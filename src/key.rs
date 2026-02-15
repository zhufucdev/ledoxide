use axum::{RequestPartsExt, extract::FromRequestParts};
use axum_extra::{
    TypedHeader,
    headers::{Authorization, authorization::Bearer},
};
use rand::{RngExt, rngs::StdRng};

use crate::{error, state::AppState};

pub struct ValidKey {}

impl FromRequestParts<AppState> for ValidKey {
    type Rejection = error::AuthError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await?;
        if bearer.token() == state.auth_key() {
            const VALID_KEY: ValidKey = ValidKey {};
            Ok(VALID_KEY)
        } else {
            Err(error::AuthError::InvalidKey)
        }
    }
}

pub fn generate_random_key() -> String {
    let mut rng: StdRng = rand::make_rng();
    const DICT: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_(";
    String::from_utf8(
        (0..=32)
            .map(|_| DICT[rng.random::<i32>() as usize % DICT.len()])
            .collect(),
    )
    .unwrap()
}
