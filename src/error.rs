use axum::{Json, http::StatusCode, response::IntoResponse};
use axum_extra::typed_header::TypedHeaderRejection;
use clap::error;
use image::ImageError;
use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, GrammarError, LlamaContextLoadError,
    LlamaModelLoadError, TokenToStringError,
    llama_batch::BatchAddError,
    mtmd::{MtmdEvalError, MtmdInitError, MtmdTokenizeError},
};
use serde_json::json;
use strum::Display;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("invalid key")]
    InvalidKey,
    #[error("invalid request header")]
    InvalidRequestHeader,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            AuthError::InvalidKey => StatusCode::UNAUTHORIZED,
            AuthError::InvalidRequestHeader => StatusCode::BAD_REQUEST,
        };
        let body = Json(json!({
            "error": self.to_string(),
        }));
        (status, body).into_response()
    }
}

impl From<TypedHeaderRejection> for AuthError {
    fn from(_: TypedHeaderRejection) -> Self {
        Self::InvalidRequestHeader
    }
}

#[derive(Debug, Display)]
pub enum CreateTaskError {
    #[strum(to_string = "invalid request: {0}")]
    InvalidRequest(anyhow::Error),
    #[strum(to_string = "missing field: {0}")]
    MissingField(String),
    #[strum(to_string = "unknown field: {0}")]
    UnknownField(String),
    #[strum(to_string = "invalid field: {0}")]
    InvalidField(String),
}

impl IntoResponse for CreateTaskError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(json!({ "error": self.to_string()}));
        let status = match self {
            CreateTaskError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            CreateTaskError::MissingField(_) => StatusCode::BAD_REQUEST,
            CreateTaskError::UnknownField(_) => StatusCode::BAD_REQUEST,
            CreateTaskError::InvalidField(_) => StatusCode::BAD_REQUEST,
        };
        (status, body).into_response()
    }
}

impl<E> From<E> for CreateTaskError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(value: E) -> Self {
        Self::InvalidRequest(value.into())
    }
}

#[derive(Debug, Error)]
pub enum RunTaskError {
    #[error("{0}")]
    Generic(#[from] anyhow::Error),
    #[error("runner: {0}")]
    Runner(#[from] RunnerError),
    #[error("empty amount, model responded with {0}")]
    EmptyAmount(String),
    #[error("invalid image in request: {0}")]
    InvalidInputImage(#[from] ImageError),
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("load context: {0}")]
    LoadContext(#[from] LlamaContextLoadError),
    #[error("load chat template: {0}")]
    LoadChatTemplate(#[from] ChatTemplateError),
    #[error("apply chat template: {0}")]
    ApplyChatTemplate(#[from] ApplyChatTemplateError),
    #[error("mtmd tokenize: {0}")]
    MtmdTokenize(#[from] MtmdTokenizeError),
    #[error("token-string conversion: {0}")]
    RunTask(#[from] TokenToStringError),
    #[error("batch add: {0}")]
    BatchAdd(#[from] BatchAddError),
    #[error("mtmd eval: {0}")]
    MtmdEval(#[from] MtmdEvalError),
    #[error("batch decode: {0}")]
    BatchDecode(#[from] DecodeError),
    #[error("llguidance: {0}")]
    Llguidance(#[from] GrammarError),
}

#[derive(Debug, Error)]
pub enum GetTaskError {
    #[error("task not found")]
    NotFound,
    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for GetTaskError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            GetTaskError::NotFound => StatusCode::NOT_FOUND,
            GetTaskError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({
                    "error": self.to_string(),
        }));
        (status, body).into_response()
    }
}

#[derive(Debug, Error)]
pub enum CreateLlamaCppRunnerError {
    #[error("hf hub: {0}")]
    HfHub(#[from] hf_hub::api::tokio::ApiError),
    #[error("load model: {0}")]
    LoadModel(#[from] LlamaModelLoadError),
    #[error("load mtmd: {0}")]
    LoadMtmd(#[from] MtmdInitError),
    #[error("load chat template: {0}")]
    LoadChatTemplate(#[from] ChatTemplateError),
}
