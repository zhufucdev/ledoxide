use std::borrow::Cow;
use std::io::{Cursor, Read as _};

#[cfg(feature = "openai")]
use async_openai::Client;
#[cfg(feature = "openai")]
use async_openai::config::OpenAIConfig;

use axum::body::Bytes;
use axum_extra::headers::Mime;
#[cfg(feature = "ollama")]
use ollama_rs::Ollama;
use postcard::from_bytes;
use zip::ZipArchive;
use zip::result::ZipError;

use crate::error::CreateTaskError;

pub trait FromEnvVars {
    fn from_env_vars() -> Self;
}

#[cfg(feature = "ollama")]
impl FromEnvVars for Ollama {
    fn from_env_vars() -> Self {
        match std::env::var("OLLAMA_BASE_URL").or_else(|_| std::env::var("OLLAMA_HOST")) {
            Ok(url) => Ollama::try_new(&url).unwrap_or_else(|_| Ollama::default()),
            Err(_) => Ollama::default(),
        }
    }
}

#[cfg(feature = "openai")]
impl FromEnvVars for Client<OpenAIConfig> {
    fn from_env_vars() -> Self {
        Client::default()
    }
}

pub trait ExtractImageBuf {
    type Error;
    fn extract_image_buf(self) -> Result<Box<[Box<[u8]>]>, Self::Error>;
}

impl ExtractImageBuf for (Bytes, Mime) {
    type Error = CreateTaskError;

    fn extract_image_buf(self) -> Result<Box<[Box<[u8]>]>, Self::Error> {
        let (source, mime) = self;
        if mime.type_() == "image" {
            return Ok(vec![Box::<[u8]>::from(source.to_vec())].into());
        } else if mime.type_() != "application" {
            return Err(CreateTaskError::UnspecificContentType(mime.to_string()));
        }
        let mut bufs = Vec::new();
        match mime.subtype().as_str() {
            "zip" | "zip-compressed" => {
                let mut archive = ZipArchive::new(Cursor::new(source))?;
                for i in 0..archive.len() {
                    let item = archive.by_index(i)?;
                    if item.is_file() {
                        bufs.push(item.bytes().collect::<Result<Box<_>, _>>()?);
                    } else {
                        return Err(ZipError::InvalidArchive(Cow::Owned(
                            "accept files only, got dir / symlink".into(),
                        ))
                        .into());
                    }
                }
            }
            _ => return Err(CreateTaskError::UnsupportedFileType(mime.to_string())),
        }
        Ok(bufs.into())
    }
}
