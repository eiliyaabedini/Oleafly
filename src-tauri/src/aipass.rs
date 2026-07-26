//! Native AI Pass OAuth and authenticated AI transport.
//!
//! OAuth codes, PKCE material, access tokens, and refresh tokens never cross
//! Tauri IPC. The webview receives only connection/profile metadata, discovered
//! model metadata, and bounded chat response bytes. Long-lived tokens are one
//! encrypted secret snapshot so refresh rotation is atomic.

use axum::{
    body::Body,
    extract::{Query, State as AxumState},
    http::{header, HeaderValue, StatusCode},
    response::Response,
    routing::get,
    Router,
};
use base64::Engine;
use futures_util::StreamExt;
use rand::RngCore;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;
use tauri::{ipc::Channel, AppHandle, State as TauriState};
use tauri_plugin_shell::ShellExt;
use tokio::sync::{oneshot, Mutex};

use crate::{secrets, state::AppState};

const ISSUER: &str = "https://aipass.one";
const METADATA_URL: &str = "https://aipass.one/.well-known/oauth-authorization-server";
const MODELS_URL: &str = "https://aipass.one/oauth2/v1/models";
const CHAT_URL: &str = "https://aipass.one/oauth2/v1/chat/completions";
const TOKEN_ACCOUNT: &str = "aipass_oauth_tokens";
const OAUTH_SCOPE: &str = "api:access profile:read";
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(30);
const CHAT_TIMEOUT: Duration = Duration::from_secs(300);
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_CONTROL_BYTES: usize = 256 * 1024;
const MAX_ERROR_BYTES: usize = 64 * 1024;
const MAX_STREAM_BYTES: usize = 16 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOKEN_BYTES: usize = 16 * 1024;

#[derive(Clone, Deserialize)]
struct OAuthMetadata {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: String,
    revocation_endpoint: String,
    #[serde(default)]
    scopes_supported: Vec<String>,
    #[serde(default)]
    response_types_supported: Vec<String>,
    #[serde(default)]
    grant_types_supported: Vec<String>,
    #[serde(default)]
    code_challenge_methods_supported: Vec<String>,
    #[serde(default)]
    token_endpoint_auth_methods_supported: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiPassProfile {
    pub subject: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AiPassStatus {
    pub available: bool,
    pub connected: bool,
    pub profile: Option<AiPassProfile>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AiPassModel {
    pub id: String,
    pub name: String,
    pub supports_vision: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct StoredTokens {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    scope: String,
    profile: Option<AiPassProfile>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default = "default_expires_in")]
    expires_in: u64,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default = "default_token_type")]
    token_type: String,
}

fn default_expires_in() -> u64 {
    3600
}

fn default_token_type() -> String {
    "Bearer".into()
}

struct PkceMaterial {
    state: String,
    verifier: String,
    challenge: String,
}

struct CallbackConfig {
    redirect_uri: String,
    address: SocketAddr,
    path: String,
}

struct ClientConfig {
    client_id: String,
    callback: CallbackConfig,
}

type CallbackSender = oneshot::Sender<Result<String, String>>;

#[derive(Clone)]
struct CallbackState {
    expected_state: Arc<String>,
    result: Arc<Mutex<Option<CallbackSender>>>,
}

#[derive(Deserialize)]
struct CallbackQuery {
    state: Option<String>,
    code: Option<String>,
    error: Option<String>,
}

#[derive(Serialize)]
pub struct DisconnectResult {
    pub revoked: bool,
}

#[derive(Serialize)]
pub struct NativeResponseMeta {
    pub status: u16,
    pub content_type: String,
}

#[derive(Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum NativeStreamEvent {
    Chunk { data: String },
    End,
    Error { message: String },
}

// The release maintainer supplies their own public OAuth client ID through
// protected build/runtime configuration; no deployment client ID is embedded.
fn protected_client_id() -> Option<String> {
    std::env::var("AIPASS_OAUTH_CLIENT_ID")
        .ok()
        .or_else(|| option_env!("AIPASS_OAUTH_CLIENT_ID").map(str::to_owned))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && value.len() <= 512)
}

// This must exactly match the redirect URI registered for that OAuth client.
fn protected_callback_uri() -> Option<String> {
    std::env::var("AIPASS_OAUTH_REDIRECT_URI")
        .ok()
        .or_else(|| option_env!("AIPASS_OAUTH_REDIRECT_URI").map(str::to_owned))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && value.len() <= 1024)
}

fn client_config() -> Result<ClientConfig, String> {
    let client_id = protected_client_id()
        .ok_or_else(|| "AI Pass is not configured in this build.".to_string())?;
    let redirect_uri = protected_callback_uri()
        .ok_or_else(|| "AI Pass callback is not configured in this build.".to_string())?;
    let callback = parse_callback(&redirect_uri)?;
    Ok(ClientConfig {
        client_id,
        callback,
    })
}

fn configuration_available() -> bool {
    client_config().is_ok()
}

fn parse_callback(raw: &str) -> Result<CallbackConfig, String> {
    let url = reqwest::Url::parse(raw)
        .map_err(|_| "AI Pass callback configuration is invalid.".to_string())?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.port().is_none()
        || url.port() == Some(0)
        || url.path().is_empty()
        || url.path() == "/"
        || url.path().contains('%')
    {
        return Err("AI Pass callback must be an explicit 127.0.0.1 loopback HTTP URL.".into());
    }
    let port = url
        .port()
        .ok_or_else(|| "AI Pass callback port is missing.".to_string())?;
    Ok(CallbackConfig {
        redirect_uri: url.to_string(),
        address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
        path: url.path().to_owned(),
    })
}

fn pinned_endpoint(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw)
        .map_err(|_| "AI Pass OAuth metadata contains an invalid endpoint.".to_string())?;
    if url.scheme() != "https"
        || url.host_str() != Some("aipass.one")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port_or_known_default() != Some(443)
        || url.fragment().is_some()
    {
        return Err("AI Pass OAuth metadata endpoint is outside the pinned origin.".into());
    }
    Ok(url)
}

fn validate_metadata(metadata: &OAuthMetadata) -> Result<(), String> {
    if metadata.issuer.trim_end_matches('/') != ISSUER {
        return Err("AI Pass OAuth metadata issuer mismatch.".into());
    }
    for endpoint in [
        &metadata.authorization_endpoint,
        &metadata.token_endpoint,
        &metadata.userinfo_endpoint,
        &metadata.revocation_endpoint,
    ] {
        pinned_endpoint(endpoint)?;
    }
    if !metadata
        .code_challenge_methods_supported
        .iter()
        .any(|method| method == "S256")
    {
        return Err("AI Pass does not advertise PKCE S256.".into());
    }
    if !metadata
        .token_endpoint_auth_methods_supported
        .iter()
        .any(|method| method == "none")
    {
        return Err("AI Pass does not advertise public-client token exchange.".into());
    }
    if !metadata
        .response_types_supported
        .iter()
        .any(|response| response == "code")
        || !metadata
            .grant_types_supported
            .iter()
            .any(|grant| grant == "authorization_code")
        || !metadata
            .grant_types_supported
            .iter()
            .any(|grant| grant == "refresh_token")
    {
        return Err("AI Pass OAuth metadata does not support the required grants.".into());
    }
    for scope in ["api:access", "profile:read"] {
        if !metadata.scopes_supported.iter().any(|item| item == scope) {
            return Err("AI Pass OAuth metadata does not support the required scopes.".into());
        }
    }
    Ok(())
}

fn http_client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent("Oleafly")
        .connect_timeout(Duration::from_secs(10))
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Could not initialize secure AI Pass transport.".to_string())
}

async fn read_bounded_body(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err("AI Pass response exceeded the size limit.".into());
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(next) = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next())
        .await
        .map_err(|_| "AI Pass response timed out.".to_string())?
    {
        let chunk = next.map_err(|_| "AI Pass response stream failed.".to_string())?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err("AI Pass response exceeded the size limit.".into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn parse_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    max_bytes: usize,
    operation: &str,
) -> Result<T, String> {
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{operation} failed (HTTP {}).", status.as_u16()));
    }
    let body = read_bounded_body(response, max_bytes).await?;
    serde_json::from_slice(&body).map_err(|_| format!("{operation} returned invalid JSON."))
}

async fn load_metadata() -> Result<OAuthMetadata, String> {
    let response = http_client(CONTROL_TIMEOUT)?
        .get(METADATA_URL)
        .send()
        .await
        .map_err(|_| "Could not reach AI Pass OAuth metadata.".to_string())?;
    let metadata =
        parse_json_response::<OAuthMetadata>(response, MAX_CONTROL_BYTES, "AI Pass metadata")
            .await?;
    validate_metadata(&metadata)?;
    Ok(metadata)
}

fn random_urlsafe(bytes: usize) -> Result<String, String> {
    let mut value = vec![0u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut value);
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value))
}

fn pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

fn generate_pkce() -> Result<PkceMaterial, String> {
    let state = random_urlsafe(32)?;
    let verifier = random_urlsafe(64)?;
    let challenge = pkce_challenge(&verifier);
    Ok(PkceMaterial {
        state,
        verifier,
        challenge,
    })
}

fn authorization_url(
    metadata: &OAuthMetadata,
    client: &ClientConfig,
    pkce: &PkceMaterial,
) -> Result<String, String> {
    let mut url = pinned_endpoint(&metadata.authorization_endpoint)?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &client.client_id)
        .append_pair("redirect_uri", &client.callback.redirect_uri)
        .append_pair("scope", OAUTH_SCOPE)
        .append_pair("state", &pkce.state)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(url.to_string())
}

fn constant_time_equal(left: &str, right: &str) -> bool {
    left.as_bytes().ct_eq(right.as_bytes()).into()
}

fn callback_response(status: StatusCode, message: &'static str) -> Response {
    let mut response = Response::new(Body::from(message));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'none'; frame-ancestors 'none'"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response
        .headers_mut()
        .insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    response.headers_mut().insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    response
}

async fn oauth_callback(
    AxumState(state): AxumState<CallbackState>,
    Query(query): Query<CallbackQuery>,
) -> Response {
    let Some(received_state) = query.state.as_deref() else {
        return callback_response(StatusCode::BAD_REQUEST, "Missing OAuth state.");
    };
    if !constant_time_equal(received_state, &state.expected_state) {
        return callback_response(StatusCode::UNAUTHORIZED, "Invalid OAuth state.");
    }

    let result = if query.error.is_some() {
        Err("AI Pass authorization was cancelled.".to_string())
    } else if let Some(code) = query
        .code
        .filter(|value| !value.is_empty() && value.len() <= 4096)
    {
        Ok(code)
    } else {
        Err("AI Pass authorization did not return a code.".to_string())
    };
    let success = result.is_ok();
    if let Some(sender) = state.result.lock().await.take() {
        let _ = sender.send(result);
    }
    if success {
        callback_response(
            StatusCode::OK,
            "AI Pass connected. You can close this window and return to Oleafly.",
        )
    } else {
        callback_response(
            StatusCode::BAD_REQUEST,
            "AI Pass authorization was not completed. You can close this window.",
        )
    }
}

#[allow(deprecated)]
fn open_system_browser(app: &AppHandle, url: String) -> Result<(), String> {
    app.shell()
        .open(url, None)
        .map_err(|_| "Could not open the system browser for AI Pass.".to_string())
}

async fn receive_authorization_code(
    app: &AppHandle,
    metadata: &OAuthMetadata,
    client: &ClientConfig,
    pkce: &PkceMaterial,
) -> Result<String, String> {
    let listener = tokio::net::TcpListener::bind(client.callback.address)
        .await
        .map_err(|_| "Could not bind the configured AI Pass loopback callback.".to_string())?;
    let url = authorization_url(metadata, client, pkce)?;
    let (result_tx, result_rx) = oneshot::channel();
    let callback_state = CallbackState {
        expected_state: Arc::new(pkce.state.clone()),
        result: Arc::new(Mutex::new(Some(result_tx))),
    };
    let router = Router::new()
        .route(&client.callback.path, get(oauth_callback))
        .with_state(callback_state);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = tauri::async_runtime::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    });

    if let Err(error) = open_system_browser(app, url) {
        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(2), server).await;
        return Err(error);
    }

    let result = tokio::time::timeout(CALLBACK_TIMEOUT, result_rx).await;
    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(2), server).await;
    match result {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err("AI Pass callback closed before completion.".into()),
        Err(_) => Err("AI Pass authorization timed out.".into()),
    }
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn validate_token(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > MAX_TOKEN_BYTES {
        return Err(format!("AI Pass returned an invalid {label}."));
    }
    Ok(())
}

fn merge_token_response(
    existing: &StoredTokens,
    response: TokenResponse,
    now: u64,
) -> Result<StoredTokens, String> {
    if !response.token_type.eq_ignore_ascii_case("bearer") {
        return Err("AI Pass returned an unsupported token type.".into());
    }
    validate_token(&response.access_token, "access token")?;
    if let Some(refresh) = response.refresh_token.as_deref() {
        validate_token(refresh, "refresh token")?;
    }
    if response.scope.as_deref().is_some_and(|scope| {
        !scope
            .split_ascii_whitespace()
            .any(|item| item == "api:access")
    }) {
        return Err("AI Pass token is missing the required API scope.".into());
    }
    let refresh_token = response
        .refresh_token
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| existing.refresh_token.clone());
    validate_token(&refresh_token, "refresh token")?;
    let expires_in = response.expires_in.clamp(60, 60 * 60 * 24 * 30);
    Ok(StoredTokens {
        access_token: response.access_token,
        refresh_token,
        expires_at: now.saturating_add(expires_in),
        scope: response.scope.unwrap_or_else(|| existing.scope.clone()),
        profile: existing.profile.clone(),
    })
}

fn read_tokens() -> Result<Option<StoredTokens>, String> {
    let Some(raw) = secrets::get_secret(TOKEN_ACCOUNT)? else {
        return Ok(None);
    };
    let tokens: StoredTokens =
        serde_json::from_str(&raw).map_err(|_| "Stored AI Pass session is corrupt.".to_string())?;
    validate_token(&tokens.access_token, "stored access token")?;
    validate_token(&tokens.refresh_token, "stored refresh token")?;
    Ok(Some(tokens))
}

fn write_tokens(tokens: &StoredTokens) -> Result<(), String> {
    let snapshot = serde_json::to_string(tokens)
        .map_err(|_| "Could not serialize the AI Pass session.".to_string())?;
    secrets::set_secret(TOKEN_ACCOUNT, &snapshot)
}

fn replace_tokens_for_connection(tokens: &StoredTokens) -> Result<Option<StoredTokens>, String> {
    let previous = read_tokens()?;
    write_tokens(tokens)?;
    Ok(previous)
}

fn clear_tokens() -> Result<(), String> {
    secrets::set_secret(TOKEN_ACCOUNT, "")
}

fn take_tokens_for_disconnect() -> Result<Option<StoredTokens>, String> {
    let tokens = read_tokens()?;
    clear_tokens()?;
    Ok(tokens)
}

pub fn local_status() -> Result<AiPassStatus, String> {
    let tokens = read_tokens()?;
    Ok(AiPassStatus {
        available: configuration_available(),
        connected: tokens.is_some(),
        profile: tokens.and_then(|value| value.profile),
    })
}

async fn request_tokens(
    metadata: &OAuthMetadata,
    fields: &[(&str, &str)],
) -> Result<TokenResponse, String> {
    let response = http_client(CONTROL_TIMEOUT)?
        .post(pinned_endpoint(&metadata.token_endpoint)?)
        .header(header::ACCEPT, "application/json")
        .form(fields)
        .send()
        .await
        .map_err(|_| "Could not reach the AI Pass token endpoint.".to_string())?;
    parse_json_response(response, MAX_CONTROL_BYTES, "AI Pass token exchange").await
}

async fn exchange_code(
    metadata: &OAuthMetadata,
    client: &ClientConfig,
    code: &str,
    verifier: &str,
) -> Result<StoredTokens, String> {
    let response = request_tokens(
        metadata,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", client.client_id.as_str()),
            ("code", code),
            ("redirect_uri", client.callback.redirect_uri.as_str()),
            ("code_verifier", verifier),
        ],
    )
    .await?;
    let empty = StoredTokens {
        access_token: String::new(),
        refresh_token: String::new(),
        expires_at: 0,
        scope: OAUTH_SCOPE.into(),
        profile: None,
    };
    merge_token_response(&empty, response, now_epoch_seconds())
}

async fn fetch_profile(
    metadata: &OAuthMetadata,
    access_token: &str,
) -> Result<Option<AiPassProfile>, String> {
    let response = http_client(CONTROL_TIMEOUT)?
        .get(pinned_endpoint(&metadata.userinfo_endpoint)?)
        .bearer_auth(access_token)
        .header(header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| "Could not reach AI Pass user information.".to_string())?;
    let value: Value =
        parse_json_response(response, MAX_CONTROL_BYTES, "AI Pass user information").await?;
    let object = value
        .as_object()
        .ok_or_else(|| "AI Pass user information was invalid.".to_string())?;
    let pick = |names: &[&str]| {
        names.iter().find_map(|name| {
            object
                .get(*name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty() && value.len() <= 512)
                .map(str::to_owned)
        })
    };
    let subject = pick(&["sub", "id"]).unwrap_or_default();
    let display_name = pick(&["name", "preferred_username"]);
    let email = pick(&["email"]);
    if subject.is_empty() && display_name.is_none() && email.is_none() {
        return Ok(None);
    }
    Ok(Some(AiPassProfile {
        subject,
        display_name,
        email,
    }))
}

async fn refresh_tokens(
    metadata: &OAuthMetadata,
    client_id: &str,
    existing: &StoredTokens,
) -> Result<StoredTokens, String> {
    let response = request_tokens(
        metadata,
        &[
            ("grant_type", "refresh_token"),
            ("client_id", client_id),
            ("refresh_token", existing.refresh_token.as_str()),
        ],
    )
    .await?;
    let rotated = merge_token_response(existing, response, now_epoch_seconds())?;
    // Access and rotated refresh tokens are one encrypted value and one
    // atomic secret-store replacement. Never use the response if persistence
    // fails, because that would lose the only valid rotated refresh token.
    write_tokens(&rotated)?;
    Ok(rotated)
}

async fn valid_access_token(state: &AppState) -> Result<String, String> {
    let _guard = state.aipass_auth_lock.lock().await;
    let tokens = read_tokens()?.ok_or_else(|| "Connect AI Pass in Settings first.".to_string())?;
    if tokens.expires_at > now_epoch_seconds().saturating_add(60) {
        return Ok(tokens.access_token);
    }
    let client = client_config()?;
    let metadata = load_metadata().await?;
    Ok(refresh_tokens(&metadata, &client.client_id, &tokens)
        .await?
        .access_token)
}

async fn refresh_after_unauthorized(
    state: &AppState,
    failed_access_token: &str,
) -> Result<String, String> {
    let _guard = state.aipass_auth_lock.lock().await;
    let tokens = read_tokens()?.ok_or_else(|| "Connect AI Pass in Settings first.".to_string())?;
    if tokens.access_token != failed_access_token
        && tokens.expires_at > now_epoch_seconds().saturating_add(60)
    {
        return Ok(tokens.access_token);
    }
    let client = client_config()?;
    let metadata = load_metadata().await?;
    Ok(refresh_tokens(&metadata, &client.client_id, &tokens)
        .await?
        .access_token)
}

#[tauri::command]
pub fn aipass_status() -> Result<AiPassStatus, String> {
    local_status()
}

#[tauri::command]
pub async fn aipass_connect(
    app: AppHandle,
    state: TauriState<'_, AppState>,
) -> Result<AiPassStatus, String> {
    let _connect_guard = state.aipass_connect_lock.lock().await;
    let client = client_config()?;
    let metadata = load_metadata().await?;
    let pkce = generate_pkce()?;
    let code = receive_authorization_code(&app, &metadata, &client, &pkce).await?;
    let mut tokens = exchange_code(&metadata, &client, &code, &pkce.verifier).await?;
    tokens.profile = fetch_profile(&metadata, &tokens.access_token)
        .await
        .unwrap_or(None);
    let persist_result = {
        let _auth_guard = state.aipass_auth_lock.lock().await;
        replace_tokens_for_connection(&tokens)
    };
    let previous = match persist_result {
        Ok(previous) => previous,
        Err(error) => {
            let _ = revoke_token(
                &metadata,
                Some(&client.client_id),
                &tokens.access_token,
                "access_token",
            )
            .await;
            let _ = revoke_token(
                &metadata,
                Some(&client.client_id),
                &tokens.refresh_token,
                "refresh_token",
            )
            .await;
            return Err(error);
        }
    };
    if let Some(previous) = previous.filter(|previous| {
        previous.access_token != tokens.access_token
            || previous.refresh_token != tokens.refresh_token
    }) {
        let _ = revoke_token(
            &metadata,
            Some(&client.client_id),
            &previous.access_token,
            "access_token",
        )
        .await;
        let _ = revoke_token(
            &metadata,
            Some(&client.client_id),
            &previous.refresh_token,
            "refresh_token",
        )
        .await;
    }
    local_status()
}

async fn revoke_token(
    metadata: &OAuthMetadata,
    client_id: Option<&str>,
    token: &str,
    hint: &str,
) -> bool {
    let mut fields = vec![("token", token), ("token_type_hint", hint)];
    if let Some(client_id) = client_id {
        fields.push(("client_id", client_id));
    }
    let Ok(client) = http_client(CONTROL_TIMEOUT) else {
        return false;
    };
    let Ok(endpoint) = pinned_endpoint(&metadata.revocation_endpoint) else {
        return false;
    };
    client
        .post(endpoint)
        .header(header::ACCEPT, "application/json")
        .form(&fields)
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
}

async fn cancel_all_requests(state: &AppState) {
    let pending = {
        let mut requests = state.aipass_requests.lock().await;
        requests
            .drain()
            .map(|(_, sender)| sender)
            .collect::<Vec<_>>()
    };
    for sender in pending {
        let _ = sender.send(());
    }
}

#[tauri::command]
pub async fn aipass_disconnect(
    state: TauriState<'_, AppState>,
) -> Result<DisconnectResult, String> {
    let _connect_guard = state.aipass_connect_lock.lock().await;
    cancel_all_requests(&state).await;
    // Remove the native session before any network work. A slow or unavailable
    // metadata/revocation endpoint must not leave a locally usable bearer token
    // behind, and new chat requests fail closed while revocation is attempted.
    let tokens = {
        let _auth_guard = state.aipass_auth_lock.lock().await;
        take_tokens_for_disconnect()?
    };
    let metadata = load_metadata().await.ok();
    let client_id = protected_client_id();
    let mut revoked = tokens.is_none();
    if let (Some(tokens), Some(metadata)) = (tokens.as_ref(), metadata.as_ref()) {
        let access_revoked = revoke_token(
            metadata,
            client_id.as_deref(),
            &tokens.access_token,
            "access_token",
        )
        .await;
        let refresh_revoked = revoke_token(
            metadata,
            client_id.as_deref(),
            &tokens.refresh_token,
            "refresh_token",
        )
        .await;
        revoked = access_revoked && refresh_revoked;
    }
    // The result lets the UI explain that remote revocation could not be
    // confirmed without retaining either token locally.
    Ok(DisconnectResult { revoked })
}

fn model_name(value: &serde_json::Map<String, Value>, id: &str) -> String {
    ["name", "display_name"]
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .filter(|name| !name.is_empty() && name.len() <= 256)
        .unwrap_or(id)
        .to_owned()
}

fn model_supports_vision(value: &serde_json::Map<String, Value>) -> bool {
    let capability = value.get("capabilities");
    value
        .get("supports_vision")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || value
            .get("vision")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || capability
            .and_then(|item| item.get("vision"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || capability
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("vision")))
        || value
            .get("input_modalities")
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("image")))
}

fn model_supports_chat(value: &serde_json::Map<String, Value>) -> bool {
    if let Some(methods) = value.get("methods") {
        return methods.as_array().is_some_and(|items| {
            items
                .iter()
                .any(|item| item.as_str() == Some("chat_completions"))
        });
    }
    value
        .get("capabilities")
        .and_then(|item| item.get("chat"))
        .and_then(Value::as_bool)
        != Some(false)
}

fn validate_model_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.trim().is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
        return Err("AI Pass model discovery returned an invalid model identifier.".into());
    }
    Ok(())
}

fn parse_models(value: &Value) -> Result<Vec<AiPassModel>, String> {
    let (entries, legacy_array) = match value {
        Value::Array(entries) => (entries, true),
        Value::Object(object) if object.get("object").and_then(Value::as_str) == Some("list") => (
            object
                .get("data")
                .and_then(Value::as_array)
                .ok_or_else(|| "AI Pass model discovery returned an invalid list.".to_string())?,
            false,
        ),
        Value::Object(_) => return Err("AI Pass model discovery returned an invalid list.".into()),
        _ => return Err("AI Pass model discovery returned an invalid response.".into()),
    };
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    for entry in entries {
        let (id, name, supports_vision) = match entry {
            Value::String(id) if legacy_array => {
                validate_model_id(id)?;
                (id.as_str(), id.to_owned(), false)
            }
            Value::Object(object) => {
                let id = object.get("id").and_then(Value::as_str).ok_or_else(|| {
                    "AI Pass model discovery returned an invalid model identifier.".to_string()
                })?;
                validate_model_id(id)?;
                if !model_supports_chat(object) {
                    continue;
                }
                (id, model_name(object, id), model_supports_vision(object))
            }
            _ => {
                return Err("AI Pass model discovery returned an invalid model entry.".into());
            }
        };
        if !seen.insert(id.to_owned()) {
            continue;
        }
        models.push(AiPassModel {
            id: id.to_owned(),
            name,
            supports_vision,
        });
    }
    if models.is_empty() {
        return Err("AI Pass did not return any usable chat models.".into());
    }
    Ok(models)
}

async fn send_models_request(access_token: &str) -> Result<reqwest::Response, String> {
    http_client(CONTROL_TIMEOUT)?
        .get(MODELS_URL)
        .bearer_auth(access_token)
        .header(header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| "Could not reach AI Pass model discovery.".to_string())
}

#[tauri::command]
pub async fn aipass_list_models(
    state: TauriState<'_, AppState>,
) -> Result<Vec<AiPassModel>, String> {
    let mut access_token = valid_access_token(&state).await?;
    let mut response = send_models_request(&access_token).await?;
    if response.status() == StatusCode::UNAUTHORIZED {
        access_token = refresh_after_unauthorized(&state, &access_token).await?;
        response = send_models_request(&access_token).await?;
    }
    let value: Value =
        parse_json_response(response, MAX_CONTROL_BYTES, "AI Pass model discovery").await?;
    parse_models(&value)
}

fn validate_chat_body(body: &[u8]) -> Result<Value, String> {
    if body.is_empty() || body.len() > MAX_REQUEST_BYTES {
        return Err("AI Pass request exceeded the size limit.".into());
    }
    let value: Value = serde_json::from_slice(body)
        .map_err(|_| "AI Pass request body is invalid JSON.".to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "AI Pass chat request must be a JSON object.".to_string())?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty() && model.len() <= 256)
        .ok_or_else(|| "AI Pass chat request is missing a valid model.".to_string())?;
    if model.chars().any(char::is_control) {
        return Err("AI Pass chat request model is invalid.".into());
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "AI Pass chat request is missing messages.".to_string())?;
    if messages.len() > 256 {
        return Err("AI Pass chat request has too many messages.".into());
    }
    if object
        .get("stream")
        .is_some_and(|stream| !stream.is_boolean())
    {
        return Err("AI Pass chat stream flag is invalid.".into());
    }
    Ok(value)
}

fn validate_request_id(request_id: &str) -> Result<(), String> {
    if request_id.is_empty()
        || request_id.len() > 64
        || !request_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
    {
        return Err("AI Pass request identifier is invalid.".into());
    }
    Ok(())
}

async fn send_chat_request(
    access_token: &str,
    body: Vec<u8>,
    cancel: &mut oneshot::Receiver<()>,
) -> Result<reqwest::Response, String> {
    let request = http_client(CHAT_TIMEOUT)?
        .post(CHAT_URL)
        .bearer_auth(access_token)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "text/event-stream, application/json")
        .body(body)
        .send();
    tokio::select! {
        _ = cancel => Err("AI Pass request cancelled.".into()),
        response = request => response.map_err(|_| "AI Pass chat request failed.".to_string()),
    }
}

async fn forward_response(
    response: reqwest::Response,
    channel: Channel<NativeStreamEvent>,
    mut cancel: oneshot::Receiver<()>,
) {
    let limit = if response.status().is_success() {
        MAX_STREAM_BYTES
    } else {
        MAX_ERROR_BYTES
    };
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        let _ = channel.send(NativeStreamEvent::Error {
            message: "AI Pass response exceeded the size limit.".into(),
        });
        return;
    }
    let started = tokio::time::Instant::now();
    let mut total = 0usize;
    let mut stream = response.bytes_stream();
    loop {
        let remaining = CHAT_TIMEOUT.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            let _ = channel.send(NativeStreamEvent::Error {
                message: "AI Pass response timed out.".into(),
            });
            return;
        }
        let wait = remaining.min(STREAM_IDLE_TIMEOUT);
        let next = tokio::select! {
            _ = &mut cancel => {
                let _ = channel.send(NativeStreamEvent::Error {
                    message: "AI Pass request cancelled.".into(),
                });
                return;
            }
            next = tokio::time::timeout(wait, stream.next()) => next,
        };
        let next = match next {
            Ok(next) => next,
            Err(_) => {
                let _ = channel.send(NativeStreamEvent::Error {
                    message: "AI Pass response timed out.".into(),
                });
                return;
            }
        };
        let Some(next) = next else {
            let _ = channel.send(NativeStreamEvent::End);
            return;
        };
        let chunk = match next {
            Ok(chunk) => chunk,
            Err(_) => {
                let _ = channel.send(NativeStreamEvent::Error {
                    message: "AI Pass response stream failed.".into(),
                });
                return;
            }
        };
        total = total.saturating_add(chunk.len());
        if total > limit {
            let _ = channel.send(NativeStreamEvent::Error {
                message: "AI Pass response exceeded the size limit.".into(),
            });
            return;
        }
        if channel
            .send(NativeStreamEvent::Chunk {
                data: base64::engine::general_purpose::STANDARD.encode(chunk),
            })
            .is_err()
        {
            return;
        }
    }
}

async fn remove_request(
    requests: &Arc<tauri::async_runtime::Mutex<HashMap<String, oneshot::Sender<()>>>>,
    request_id: &str,
) {
    requests.lock().await.remove(request_id);
}

#[tauri::command]
pub async fn aipass_chat_completions(
    request_id: String,
    body: String,
    on_event: Channel<NativeStreamEvent>,
    state: TauriState<'_, AppState>,
) -> Result<NativeResponseMeta, String> {
    validate_request_id(&request_id)?;
    let body = serde_json::to_vec(&validate_chat_body(body.as_bytes())?)
        .map_err(|_| "Could not encode AI Pass request.".to_string())?;
    let (cancel_tx, mut cancel_rx) = oneshot::channel();
    {
        let mut requests = state.aipass_requests.lock().await;
        if requests.contains_key(&request_id) {
            return Err("AI Pass request identifier is already active.".into());
        }
        requests.insert(request_id.clone(), cancel_tx);
    }

    let result = async {
        let mut access_token = valid_access_token(&state).await?;
        let mut response = send_chat_request(&access_token, body.clone(), &mut cancel_rx).await?;
        if response.status() == StatusCode::UNAUTHORIZED {
            access_token = refresh_after_unauthorized(&state, &access_token).await?;
            response = send_chat_request(&access_token, body, &mut cancel_rx).await?;
        }
        Ok::<_, String>(response)
    }
    .await;

    let response = match result {
        Ok(response) => response,
        Err(error) => {
            remove_request(&state.aipass_requests, &request_id).await;
            return Err(error);
        }
    };
    let status = response.status().as_u16();
    if !response.status().is_success() {
        // Never forward an upstream error body to the webview: a remote service
        // could reflect request headers or other sensitive diagnostics. The
        // status code is enough for the AI SDK and this bounded OpenAI-shaped
        // body contains no OAuth material or user request content.
        drop(response);
        let safe_error = serde_json::json!({
            "error": {
                "message": format!("AI Pass request failed (HTTP {status})."),
                "type": "aipass_http_error"
            }
        })
        .to_string();
        let _ = on_event.send(NativeStreamEvent::Chunk {
            data: base64::engine::general_purpose::STANDARD.encode(safe_error),
        });
        let _ = on_event.send(NativeStreamEvent::End);
        remove_request(&state.aipass_requests, &request_id).await;
        return Ok(NativeResponseMeta {
            status,
            content_type: "application/json".into(),
        });
    }
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .filter(|value| value.len() <= 128)
        .unwrap_or("application/octet-stream")
        .to_owned();
    let requests = state.aipass_requests.clone();
    let task_request_id = request_id.clone();
    tauri::async_runtime::spawn(async move {
        forward_response(response, on_event, cancel_rx).await;
        remove_request(&requests, &task_request_id).await;
    });
    Ok(NativeResponseMeta {
        status,
        content_type,
    })
}

#[tauri::command]
pub async fn aipass_cancel_request(
    request_id: String,
    state: TauriState<'_, AppState>,
) -> Result<bool, String> {
    validate_request_id(&request_id)?;
    let sender = state.aipass_requests.lock().await.remove(&request_id);
    Ok(sender.is_some_and(|sender| sender.send(()).is_ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_metadata() -> OAuthMetadata {
        serde_json::from_value(json!({
            "issuer": "https://aipass.one",
            "authorization_endpoint": "https://aipass.one/oauth2/authorize",
            "token_endpoint": "https://aipass.one/oauth2/token",
            "userinfo_endpoint": "https://aipass.one/oauth2/userinfo",
            "revocation_endpoint": "https://aipass.one/oauth2/revoke",
            "scopes_supported": ["api:access", "profile:read"],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256", "plain"],
            "token_endpoint_auth_methods_supported": ["none"]
        }))
        .unwrap()
    }

    #[test]
    fn metadata_requires_pinned_issuer_s256_and_public_client_auth() {
        assert!(validate_metadata(&valid_metadata()).is_ok());

        let mut wrong_issuer = valid_metadata();
        wrong_issuer.issuer = "https://example.com".into();
        assert!(validate_metadata(&wrong_issuer).is_err());

        let mut no_s256 = valid_metadata();
        no_s256.code_challenge_methods_supported = vec!["plain".into()];
        assert!(validate_metadata(&no_s256).is_err());

        let mut confidential_only = valid_metadata();
        confidential_only.token_endpoint_auth_methods_supported = vec!["client_secret_post".into()];
        assert!(validate_metadata(&confidential_only).is_err());
    }

    #[test]
    fn metadata_rejects_bearer_redirects_off_the_pinned_origin() {
        let mut metadata = valid_metadata();
        metadata.token_endpoint = "https://attacker.example/token".into();
        assert!(validate_metadata(&metadata).is_err());

        let mut metadata = valid_metadata();
        metadata.userinfo_endpoint = "http://aipass.one/oauth2/userinfo".into();
        assert!(validate_metadata(&metadata).is_err());
    }

    #[test]
    fn pkce_material_is_random_url_safe_and_s256() {
        let first = generate_pkce().unwrap();
        let second = generate_pkce().unwrap();
        assert_ne!(first.state, second.state);
        assert_ne!(first.verifier, second.verifier);
        assert!(first.state.len() >= 43);
        assert!(first.verifier.len() >= 43);
        assert!(first
            .state
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert_eq!(pkce_challenge(&first.verifier), first.challenge);
        assert_ne!(first.verifier, first.challenge);
    }

    #[test]
    fn callback_config_is_loopback_http_with_an_explicit_port() {
        assert!(parse_callback("http://127.0.0.1:38137/oauth/callback").is_ok());
        assert!(parse_callback("https://127.0.0.1:38137/oauth/callback").is_err());
        assert!(parse_callback("http://localhost:38137/oauth/callback").is_err());
        assert!(parse_callback("http://127.0.0.1/oauth/callback").is_err());
        assert!(parse_callback("http://127.0.0.1:0/oauth/callback").is_err());
        assert!(parse_callback("http://127.0.0.1:38137/").is_err());
        assert!(parse_callback("http://user@127.0.0.1:38137/oauth/callback").is_err());
        assert!(parse_callback("http://127.0.0.1:38137/oauth/callback?code=x").is_err());
    }

    #[test]
    fn model_discovery_uses_the_default_openai_compatible_endpoint() {
        assert_eq!(MODELS_URL, "https://aipass.one/oauth2/v1/models");
    }

    #[test]
    fn model_discovery_accepts_openai_and_legacy_shapes_without_defaults() {
        let openai = parse_models(&json!({
            "object": "list",
            "data": [
                {
                    "id": "openai/live-text",
                    "name": "Live Text",
                    "object": "model",
                    "created": 1_700_000_000,
                    "owned_by": "openai",
                    "description": "A chat model",
                    "type": "text",
                    "capabilities": ["text", "streaming"],
                    "methods": ["chat_completions", "responses"],
                    "future_additive_field": {"nested": true}
                },
                {
                    "id": "live-vision",
                    "type": "multimodal",
                    "capabilities": ["text", "image", "vision"],
                    "methods": ["chat_completions", "responses"]
                },
                {
                    "id": "audio-only",
                    "type": "audio",
                    "capabilities": ["audio", "text"],
                    "methods": ["audio_speech"]
                },
                {"id": "image-only", "capabilities": {"chat": false}},
                {"id": "openai/live-text", "name": "duplicate"}
            ]
        }))
        .unwrap();
        assert_eq!(
            openai,
            vec![
                AiPassModel {
                    id: "openai/live-text".into(),
                    name: "Live Text".into(),
                    supports_vision: false,
                },
                AiPassModel {
                    id: "live-vision".into(),
                    name: "live-vision".into(),
                    supports_vision: true,
                },
            ]
        );

        let legacy = parse_models(&json!([
            "provider/model-b",
            {"id": "model-a"},
            "provider/model-b"
        ]))
        .unwrap();
        assert_eq!(
            legacy
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["provider/model-b", "model-a"]
        );
        assert!(parse_models(&json!({"data": "not-an-array"})).is_err());
        assert!(parse_models(&json!({
            "object": "not-a-list",
            "data": [{"id": "must-not-be-accepted"}]
        }))
        .is_err());
    }

    #[test]
    fn model_discovery_rejects_malformed_ids_instead_of_partially_accepting_a_list() {
        for malformed in [
            json!({"object": "list", "data": [
                {"id": "valid/model", "methods": ["chat_completions"]},
                {"id": "", "methods": ["chat_completions"]}
            ]}),
            json!({"object": "list", "data": [
                {"id": "valid/model", "methods": ["chat_completions"]},
                {"id": 42, "methods": ["chat_completions"]}
            ]}),
            json!(["valid/model", ""]),
            json!([
                "valid/model",
                {"id": null}
            ]),
            json!({"object": "list", "data": [
                {"id": "valid/model", "methods": ["chat_completions"]},
                "legacy-string-is-not-valid-in-openai-data"
            ]}),
            json!({"object": "list", "data": [
                {"id": "valid/model", "methods": ["chat_completions"]},
                {"id": "\n", "methods": ["chat_completions"]}
            ]}),
        ] {
            assert!(parse_models(&malformed).is_err(), "{malformed}");
        }
    }

    #[test]
    fn refresh_rotation_replaces_tokens_as_one_snapshot() {
        let existing = StoredTokens {
            access_token: "old-access".into(),
            refresh_token: "old-refresh".into(),
            expires_at: 10,
            scope: "api:access profile:read".into(),
            profile: Some(AiPassProfile {
                subject: "user-1".into(),
                display_name: Some("A User".into()),
                email: None,
            }),
        };
        let rotated = merge_token_response(
            &existing,
            TokenResponse {
                access_token: "new-access".into(),
                refresh_token: Some("new-refresh".into()),
                expires_in: 3600,
                scope: None,
                token_type: "Bearer".into(),
            },
            100,
        )
        .unwrap();
        assert_eq!(rotated.access_token, "new-access");
        assert_eq!(rotated.refresh_token, "new-refresh");
        assert_eq!(rotated.expires_at, 3700);
        assert_eq!(rotated.profile, existing.profile);

        let retained_refresh = merge_token_response(
            &rotated,
            TokenResponse {
                access_token: "newer-access".into(),
                refresh_token: None,
                expires_in: 1800,
                scope: None,
                token_type: "bearer".into(),
            },
            200,
        )
        .unwrap();
        assert_eq!(retained_refresh.refresh_token, "new-refresh");
    }

    #[test]
    fn disconnect_takes_and_clears_the_local_session_before_revocation() {
        let _env_guard = crate::paths::data_dir_env_lock();
        let dir = std::env::temp_dir().join(format!(
            "oleafly-aipass-disconnect-{}-{}",
            std::process::id(),
            random_urlsafe(12).unwrap()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("OLEAFLY_DATA_DIR", &dir);
        let tokens = StoredTokens {
            access_token: "access-token".into(),
            refresh_token: "refresh-token".into(),
            expires_at: 10,
            scope: OAUTH_SCOPE.into(),
            profile: None,
        };
        write_tokens(&tokens).unwrap();

        let taken = take_tokens_for_disconnect().unwrap();
        let remaining = read_tokens().unwrap();
        std::env::remove_var("OLEAFLY_DATA_DIR");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(taken, Some(tokens));
        assert!(remaining.is_none());
    }

    #[test]
    fn connecting_atomically_replaces_and_returns_the_previous_session() {
        let _env_guard = crate::paths::data_dir_env_lock();
        let dir = std::env::temp_dir().join(format!(
            "oleafly-aipass-connect-{}-{}",
            std::process::id(),
            random_urlsafe(12).unwrap()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("OLEAFLY_DATA_DIR", &dir);
        let previous = StoredTokens {
            access_token: "old-access-token".into(),
            refresh_token: "old-refresh-token".into(),
            expires_at: 10,
            scope: OAUTH_SCOPE.into(),
            profile: None,
        };
        let replacement = StoredTokens {
            access_token: "new-access-token".into(),
            refresh_token: "new-refresh-token".into(),
            expires_at: 20,
            scope: OAUTH_SCOPE.into(),
            profile: None,
        };
        write_tokens(&previous).unwrap();

        let returned = replace_tokens_for_connection(&replacement).unwrap();
        let stored = read_tokens().unwrap();
        std::env::remove_var("OLEAFLY_DATA_DIR");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(returned, Some(previous));
        assert_eq!(stored, Some(replacement));
    }

    #[test]
    fn chat_request_validation_bounds_and_requires_openai_json() {
        assert!(validate_chat_body(br#"{"model":"live","messages":[],"stream":true}"#).is_ok());
        assert!(validate_chat_body(br#"[]"#).is_err());
        assert!(validate_chat_body(br#"{"messages":[]}"#).is_err());
        assert!(validate_chat_body(br#"{"model":"","messages":[]}"#).is_err());
        assert!(validate_chat_body(&vec![b'x'; MAX_REQUEST_BYTES + 1]).is_err());
    }

    #[tokio::test]
    async fn cancellation_terminates_a_pending_upstream_response() {
        use axum::body::Bytes;
        use std::convert::Infallible;

        async fn pending_response() -> Response {
            Response::new(Body::from_stream(futures_util::stream::pending::<
                Result<Bytes, Infallible>,
            >()))
        }

        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/", get(pending_response)))
                .await
                .unwrap();
        });
        let response = http_client(CHAT_TIMEOUT)
            .unwrap()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let channel = Channel::new(|_| Ok(()));
        let forwarding = tokio::spawn(forward_response(response, channel, cancel_rx));

        cancel_tx.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), forwarding)
                .await
                .is_ok(),
            "cancellation must drop the native upstream response promptly"
        );
        server.abort();
    }
}
