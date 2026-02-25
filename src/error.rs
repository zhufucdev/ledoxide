use std::fmt::Display;

use axum::{
    Json,
    extract::{
        FromRequest,
        multipart::{MultipartError, MultipartRejection},
        rejection::{FormRejection, JsonRejection},
    },
    http::StatusCode,
    response::IntoResponse,
};
use axum_extra::typed_header::TypedHeaderRejection;
use image::ImageError;
use serde::de::DeserializeOwned;
use serde_json::json;
use strum::Display;

#[derive(Debug, Display)]
pub enum AuthError {
    #[strum(to_string = "invalid key")]
    InvalidKey,
    #[strum(to_string = "invalid request header")]
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

#[derive(Debug, Display)]
pub enum RunTaskError {
    #[strum(to_string = "{0}")]
    Generic(anyhow::Error),
    #[strum(to_string = "mistralrs: {0}")]
    Mistral(mistralrs::error::Error),
    #[strum(to_string = "empty amount, model responded with: {0}")]
    EmptyAmount(String),
    #[strum(to_string = "invalid image in request: {0}")]
    InvalidInputImage(ImageError),
}

impl From<anyhow::Error> for RunTaskError {
    fn from(value: anyhow::Error) -> Self {
        Self::Generic(value)
    }
}

impl From<mistralrs::error::Error> for RunTaskError {
    fn from(value: mistralrs::error::Error) -> Self {
        Self::Mistral(value)
    }
}

impl From<ImageError> for RunTaskError {
    fn from(value: ImageError) -> Self {
        Self::InvalidInputImage(value)
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

#[derive(Debug, Display)]
pub enum GetTaskError {
    #[strum(to_string = "task not found")]
    NotFound,
    #[strum(to_string = "{0}")]
    Internal(anyhow::Error),
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

impl From<anyhow::Error> for GetTaskError {
    fn from(value: anyhow::Error) -> Self {
        Self::Internal(value)
    }
}
