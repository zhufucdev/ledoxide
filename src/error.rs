use axum::{Json, http::StatusCode, response::IntoResponse};
use axum_extra::typed_header::TypedHeaderRejection;
use image::ImageError;
use llama_runner::error::RunnerError;
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
    #[strum(to_string = "unspecific content-type for {0}")]
    UnspecificContentType(String),
    #[strum(to_string = "unsupported file type: {0}")]
    UnsupportedFileType(String),
}

impl IntoResponse for CreateTaskError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(json!({ "error": self.to_string()}));
        let status = match self {
            _ => StatusCode::BAD_REQUEST,
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
    #[error("invalid LLM output for {0}")]
    InvalidOutput(String),
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
