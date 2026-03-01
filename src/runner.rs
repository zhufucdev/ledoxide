use std::{
    io::IsTerminal,
    num::NonZeroU32,
    path::{Path, PathBuf},
    str::FromStr,
    sync::LazyLock,
    usize,
};

use encoding_rs::{Decoder, UTF_8};
use hf_hub::api::tokio::ApiBuilder;
use llama_cpp_2::{
    LlamaContextLoadError,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    mtmd::{self, MtmdBitmap, MtmdContext, MtmdInputText},
    sampling::LlamaSampler,
};
use strum::Display;

use crate::{
    error::{CreateLlamaCppRunnerError, RunnerError},
    sample::{LlguidanceSamplingParams, SimpleSamplingParams},
};

pub trait TextLmRunner<'a> {
    type Response: Iterator<Item = Result<String, RunnerError>>;
    fn stream_lm_response(&'a self, request: TextLmRequest) -> Self::Response;
}

pub trait VisionLmRunner<'a> {
    type Response: Iterator<Item = Result<String, RunnerError>>;
    fn stream_vlm_response(&'a self, request: VisionLmRequest) -> Self::Response;
}

pub const DEFAULT_MODEL_ID: &str = "google/gemma-3-4b-it-qat-q4_0-gguf";
pub const DEFAULT_MODEL_FILENAME: &str = "gemma-3-4b-it-q4_0.gguf";
pub const DEFAULT_MULTIMODEL_FILENAME: &str = "mmproj-model-f16-4B.gguf";

#[derive(Debug, Clone)]
pub struct RunnerRequest<M> {
    pub messages: Vec<(MessageRole, M)>,
    pub sampling: SimpleSamplingParams,
    pub llguidance: Option<LlguidanceSamplingParams>,
    pub max_seq: usize,
}

impl<M> Default for RunnerRequest<M> {
    fn default() -> Self {
        Self {
            messages: vec![],
            sampling: Default::default(),
            llguidance: None,
            max_seq: usize::MAX,
        }
    }
}

pub type TextLmRequest = RunnerRequest<String>;
pub type VisionLmRequest = RunnerRequest<ImageOrText>;

pub trait TextLmRunnerExt<'a> {
    async fn get_lm_response(&'a self, request: TextLmRequest) -> Result<String, RunnerError>;
}

pub trait VisionLmRunnerExt<'a> {
    async fn get_vlm_response(&'a self, request: VisionLmRequest) -> Result<String, RunnerError>;
}

impl<'a, T> TextLmRunnerExt<'a> for T
where
    T: TextLmRunner<'a>,
{
    async fn get_lm_response(&'a self, request: TextLmRequest) -> Result<String, RunnerError> {
        self.stream_lm_response(request)
            .collect::<Result<String, _>>()
    }
}

impl<'a, T> VisionLmRunnerExt<'a> for T
where
    T: VisionLmRunner<'a>,
{
    async fn get_vlm_response(&'a self, request: VisionLmRequest) -> Result<String, RunnerError> {
        self.stream_vlm_response(request)
            .collect::<Result<String, _>>()
    }
}

#[derive(Debug, Clone, Display, PartialEq, Eq)]
pub enum MessageRole {
    #[strum(to_string = "assistant")]
    Assistant,
    #[strum(to_string = "user")]
    User,
    #[strum(to_string = "system")]
    System,
}

#[derive(Debug, Clone)]
pub enum ImageOrText {
    Text(String),
    Image(image::DynamicImage),
}

pub struct Gemma3Runner {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    mtmd_ctx: MtmdContext,
    ctx_size: NonZeroU32,
}

static LLAMA_BACKEND: LazyLock<LlamaBackend> = LazyLock::new(|| LlamaBackend::init().unwrap());

impl Gemma3Runner {
    pub async fn new(
        model_id: impl ToString,
        model_file: impl AsRef<str>,
        multimodel_file: impl AsRef<str>,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let mut repo = ApiBuilder::new()
            .with_progress(std::io::stdin().is_terminal())
            .with_token(std::env::var("HF_TOKEN").ok());
        if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
            repo = repo.with_endpoint(endpoint);
        }
        if let Ok(cache) = std::env::var("HF_HOME") {
            repo = repo.with_cache_dir(
                PathBuf::from_str(&cache).expect("HF_HOME env var is not a valid path"),
            );
        }
        let repo = repo.build()?.model(model_id.to_string());
        let model = LlamaModel::load_from_file(
            &LLAMA_BACKEND,
            repo.get(model_file.as_ref()).await?,
            &Default::default(),
        )?;

        let mtmd_ctx = MtmdContext::init_from_file(
            repo.get(multimodel_file.as_ref()).await?.to_str().unwrap(),
            &model,
            &Default::default(),
        )?;

        let chat_template = model.chat_template(None)?;

        Ok(Self {
            model,
            mtmd_ctx,
            chat_template,
            ctx_size: 10240u32.try_into().unwrap(),
        })
    }

    pub async fn default() -> Result<Self, CreateLlamaCppRunnerError> {
        Self::new(
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_FILENAME,
            DEFAULT_MULTIMODEL_FILENAME,
        )
        .await
    }

    pub fn from_files(
        model_file: impl AsRef<Path>,
        multimodel_file: impl AsRef<Path>,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, model_file, &Default::default())?;
        let mtmd_ctx = MtmdContext::init_from_file(
            multimodel_file.as_ref().as_os_str().to_str().unwrap(),
            &model,
            &Default::default(),
        )?;

        let chat_template = model.chat_template(None)?;

        Ok(Self {
            model,
            mtmd_ctx,
            chat_template,
            ctx_size: 10240u32.try_into().unwrap(),
        })
    }
}

impl Gemma3Runner {
    fn new_context_window(&self) -> Result<LlamaContext<'_>, LlamaContextLoadError> {
        self.model.new_context(
            &LLAMA_BACKEND,
            LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
        )
    }
}

impl<'a> TextLmRunner<'a> for Gemma3Runner {
    type Response = GemmaStream<'a>;

    fn stream_lm_response(&'a self, request: TextLmRequest) -> Self::Response {
        let request: VisionLmRequest = request.into();
        let ctx = self
            .new_context_window()
            .map_err(|err| RunnerError::from(err));
        GemmaStream::new(ctx, request, self)
    }
}

impl<'a> VisionLmRunner<'a> for Gemma3Runner {
    type Response = GemmaStream<'a>;

    fn stream_vlm_response(&'a self, request: VisionLmRequest) -> Self::Response {
        let ctx = self
            .new_context_window()
            .map_err(|err| RunnerError::from(err));
        GemmaStream::new(ctx, request, self)
    }
}

impl From<TextLmRequest> for VisionLmRequest {
    fn from(value: TextLmRequest) -> Self {
        Self {
            messages: value
                .messages
                .into_iter()
                .map(|(role, text)| (role, ImageOrText::Text(text)))
                .collect(),
            sampling: value.sampling,
            llguidance: value.llguidance,
            max_seq: value.max_seq,
        }
    }
}

pub struct GemmaStream<'a> {
    ctx_source: Option<Result<LlamaContext<'a>, RunnerError>>,
    ctx: Option<LlamaContext<'a>>,
    req: VisionLmRequest,
    runner: &'a Gemma3Runner,
    runtime: Option<Runtime<'a>>,
    done: bool,
}

struct Runtime<'a> {
    sampler: LlamaSampler,
    decoder: Decoder,
    batch: LlamaBatch<'a>,
    n_past: i32,
    step: usize,
}

impl<'a> GemmaStream<'a> {
    fn prepare(&mut self) -> Result<(), RunnerError> {
        // Preprocess the message, flattening media
        let media_marker = mtmd::mtmd_default_marker();
        let messages = self
            .req
            .messages
            .iter()
            .fold(
                Vec::<(MessageRole, String)>::new(),
                |mut acc, (role, message)| {
                    let text = match message {
                        ImageOrText::Text(text) => text,
                        ImageOrText::Image(_) => media_marker,
                    };
                    if let Some(last) = acc.last()
                        && last.0 == *role
                    {
                        // merge adjacent
                        let (_, adj) = acc.remove(acc.len() - 1);
                        acc.push((role.clone(), format!("{0}\n{text}", adj)));
                        acc
                    } else {
                        acc.push((role.clone(), text.to_string()));
                        acc
                    }
                },
            )
            .into_iter()
            .map(|(role, content)| LlamaChatMessage::new(role.to_string(), content))
            .collect::<Result<Vec<_>, _>>()
            .expect("message preprocessing failed");
        log::debug!(target: "gemma", "preprocessed messages: {messages:?}");

        // Aggregate images
        let formatted_prompt =
            self.runner
                .model
                .apply_chat_template(&self.runner.chat_template, &messages, true)?;
        let bitmaps = self
            .req
            .messages
            .iter()
            .filter_map(|msg| match &msg.1 {
                ImageOrText::Image(image) => Some(image),
                _ => None,
            })
            .enumerate()
            .map(|(idx, im)| {
                MtmdBitmap::from_image_data(
                    im.width(),
                    im.height(),
                    im.to_rgb8().to_vec().as_slice(),
                )
                .expect(format!("image#{} has corrupted RGB data", idx).as_str())
            })
            .collect::<Vec<_>>();
        let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = self.runner.mtmd_ctx.tokenize(
            MtmdInputText {
                text: formatted_prompt,
                add_special: true,
                parse_special: true,
            },
            &bitmap_refs,
        )?;
        log::debug!(target: "gemma", "tokenization resulted in {} chunks", chunks.len());
        let n_past = chunks.eval_chunks(
            &self.runner.mtmd_ctx,
            self.ctx.as_ref().unwrap(),
            0,
            0,
            1,
            true,
        )?;

        // Generate preparation
        let mut preparation = Runtime {
            sampler: self.req.sampling.to_llama(),
            decoder: UTF_8.new_decoder(),
            batch: LlamaBatch::new(self.runner.ctx_size.get() as usize, 1),
            n_past,
            step: 0,
        };
        if let Some(llguidance) = &self.req.llguidance {
            let llg_sampler = llguidance.to_llama(&self.runner.model)?;
            preparation.sampler = LlamaSampler::chain_simple([llg_sampler, preparation.sampler]);
        }

        self.runtime = Some(preparation);
        Ok(())
    }
}

impl Iterator for GemmaStream<'_> {
    type Item = Result<String, RunnerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if let Some(result) = self.ctx_source.take() {
            match result {
                Ok(ctx) => self.ctx = Some(ctx),
                Err(err) => {
                    self.done = true;
                    return Some(Err(err));
                }
            }
        }

        if self.runtime.is_none()
            && let Err(err) = self.prepare()
        {
            self.done = true;
            return Some(Err(err));
        }
        let Runtime {
            sampler,
            decoder,
            batch,
            n_past,
            step,
        } = self.runtime.as_mut().unwrap();

        if *step >= self.req.max_seq {
            self.done = true;
            return None;
        }

        // Sample response token
        let ctx = self.ctx.as_mut().unwrap();
        let runner = self.runner;
        let step_copy = step.clone();
        let mut sample = move || -> Result<Option<String>, RunnerError> {
            let token = (&sampler.sample(ctx, -1)).clone();
            sampler.accept(token);
            if runner.model.is_eog_token(token) {
                return Ok(None);
            }
            batch.clear();
            batch.add(token, *n_past + (step_copy as i32), &[0], true)?;

            ctx.decode(batch)?;

            let piece = runner.model.token_to_piece(token, decoder, true, None)?;
            Ok(Some(piece))
        };
        match sample() {
            Ok(Some(piece)) => {
                *step += 1;
                return Some(Ok(piece));
            }
            Ok(None) => {
                self.done = true;
                return None;
            }
            Err(err) => {
                self.done = true;
                return Some(Err(err));
            }
        }
    }
}

impl<'a> GemmaStream<'a> {
    fn new(
        source: Result<LlamaContext<'a>, RunnerError>,
        req: VisionLmRequest,
        runner: &'a Gemma3Runner,
    ) -> Self {
        Self {
            ctx_source: Some(source),
            ctx: None,
            req,
            runner,
            runtime: None,
            done: false,
        }
    }
}
