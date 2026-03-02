use std::{
    env,
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
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    mtmd::{self, MtmdBitmap, MtmdContext, MtmdInputText},
    sampling::LlamaSampler,
};
use strum::Display;

use crate::{
    error::{CreateLlamaCppRunnerError, RunnerError},
    sample::{LlguidanceSamplingParams, SimpleSamplingParams},
};

pub const QWEN_3_VL_4B_GUFF_MODEL_ID: &str = "Qwen/Qwen3-VL-4B-Instruct-GGUF";
pub const QWEN_3_VL_4B_GUFF_MODDEL_FILENAME: &str = "Qwen3VL-4B-Instruct-Q4_K_M.gguf";
pub const QWEN_3_VL_4B_GUFF_MULTIMODEL_FILENAME: &str = "mmproj-Qwen3VL-4B-Instruct-F16.gguf";

pub const GEMMA_3_1B_GUFF_MODEL_ID: &str = "google/gemma-3-1b-it-qat-q4_0-gguf";
pub const GEMMA_3_1B_GUFF_MODEL_FILENAME: &str = "gemma-3-1b-it-q4_0.gguf";

pub trait TextLmRunner<'a> {
    type Response: Iterator<Item = Result<String, RunnerError>>;
    fn stream_lm_response(&'a self, request: TextLmRequest) -> Self::Response;
}

pub trait VisionLmRunner<'a> {
    type Response: Iterator<Item = Result<String, RunnerError>>;
    fn stream_vlm_response(&'a self, request: VisionLmRequest) -> Self::Response;
}

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
    fn get_lm_response(&'a self, request: TextLmRequest) -> Result<String, RunnerError>;
}

pub trait VisionLmRunnerExt<'a> {
    fn get_vlm_response(&'a self, request: VisionLmRequest) -> Result<String, RunnerError>;
}

impl<'a, T> TextLmRunnerExt<'a> for T
where
    T: TextLmRunner<'a>,
{
    fn get_lm_response(&'a self, request: TextLmRequest) -> Result<String, RunnerError> {
        self.stream_lm_response(request)
            .collect::<Result<String, _>>()
    }
}

impl<'a, T> VisionLmRunnerExt<'a> for T
where
    T: VisionLmRunner<'a>,
{
    fn get_vlm_response(&'a self, request: VisionLmRequest) -> Result<String, RunnerError> {
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

pub struct Gemma3TextRunner {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    ctx_size: NonZeroU32,
}

impl Gemma3TextRunner {
    pub async fn new(
        model_id: impl ToString,
        model_file: impl AsRef<str>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let repo = build_hf_api()?.model(model_id.to_string());
        Self::from_file(repo.get(model_file.as_ref()).await?, ctx_size)
    }

    pub async fn default() -> Result<Self, CreateLlamaCppRunnerError> {
        Self::new(
            GEMMA_3_1B_GUFF_MODEL_ID,
            GEMMA_3_1B_GUFF_MODEL_FILENAME,
            32_000.try_into().unwrap(),
        )
        .await
    }

    pub fn from_file(
        model_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, model_file, &Default::default())?;

        let chat_template = model.chat_template(None)?;
        Ok(Self {
            model,
            chat_template,
            ctx_size,
        })
    }
}

impl<'a> TextLmRunner<'a> for Gemma3TextRunner {
    type Response = Gemma3Stream<'a, String, Gemma3TextRunner>;

    fn stream_lm_response(&'a self, request: TextLmRequest) -> Self::Response {
        let ctx = self
            .model
            .new_context(
                &LLAMA_BACKEND,
                LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
            )
            .map_err(|err| RunnerError::from(err));
        Gemma3Stream::new(ctx, request, self, &self.model)
    }
}

pub struct Gemma3VisionRunner {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    mtmd_ctx: MtmdContext,
    ctx_size: NonZeroU32,
}

static LLAMA_BACKEND: LazyLock<LlamaBackend> = LazyLock::new(|| {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(
        env::var("RUST_LOG").map_or(false, |lvl| lvl.to_lowercase() == "debug"),
    ));
    LlamaBackend::init().unwrap()
});

impl Gemma3VisionRunner {
    pub async fn new(
        repo_id: impl ToString,
        model_file: impl AsRef<str>,
        multimodel_file: impl AsRef<str>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let repo = build_hf_api()?.model(repo_id.to_string());
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
            ctx_size,
        })
    }

    pub async fn default() -> Result<Self, CreateLlamaCppRunnerError> {
        Self::new(
            QWEN_3_VL_4B_GUFF_MODEL_ID,
            QWEN_3_VL_4B_GUFF_MODDEL_FILENAME,
            QWEN_3_VL_4B_GUFF_MULTIMODEL_FILENAME,
            10240u32.try_into().unwrap(),
        )
        .await
    }

    pub fn from_files(
        model_file: impl AsRef<Path>,
        multimodel_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
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
            ctx_size,
        })
    }

    fn new_context_window(&self) -> Result<LlamaContext<'_>, LlamaContextLoadError> {
        self.model.new_context(
            &LLAMA_BACKEND,
            LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
        )
    }
}

impl<'a> TextLmRunner<'a> for Gemma3VisionRunner {
    type Response = Gemma3Stream<'a, ImageOrText, Gemma3VisionRunner>;

    fn stream_lm_response(&'a self, request: TextLmRequest) -> Self::Response {
        let request: VisionLmRequest = request.into();
        let ctx = self
            .new_context_window()
            .map_err(|err| RunnerError::from(err));
        Gemma3Stream::new(ctx, request, self, &self.model)
    }
}

impl<'a> VisionLmRunner<'a> for Gemma3VisionRunner {
    type Response = Gemma3Stream<'a, ImageOrText, Gemma3VisionRunner>;

    fn stream_vlm_response(&'a self, request: VisionLmRequest) -> Self::Response {
        let ctx = self
            .new_context_window()
            .map_err(|err| RunnerError::from(err));
        Gemma3Stream::new(ctx, request, self, &self.model)
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

pub struct Gemma3Stream<'a, M, Runner> {
    ctx_source: Option<Result<LlamaContext<'a>, RunnerError>>,
    ctx: Option<LlamaContext<'a>>,
    req: RunnerRequest<M>,
    runner: &'a Runner,
    model: &'a LlamaModel,
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

trait PrepareRun {
    fn prepare(&mut self) -> Result<(), RunnerError>;
}

impl PrepareRun for Gemma3Stream<'_, ImageOrText, Gemma3VisionRunner> {
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

impl PrepareRun for Gemma3Stream<'_, String, Gemma3TextRunner> {
    fn prepare(&mut self) -> Result<(), RunnerError> {
        // Preprocess the message
        let media_marker = mtmd::mtmd_default_marker();
        let messages = self
            .req
            .messages
            .iter()
            .fold(
                Vec::<(MessageRole, String)>::new(),
                |mut acc, (role, message)| {
                    if let Some(last) = acc.last()
                        && last.0 == *role
                    {
                        // merge adjacent
                        let (_, adj) = acc.remove(acc.len() - 1);
                        acc.push((role.clone(), format!("{0}\n{message}", adj)));
                        acc
                    } else {
                        acc.push((role.clone(), message.clone()));
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
        let token_list = self.model.str_to_token(&formatted_prompt, AddBos::Always)?;
        let mut batch = LlamaBatch::new(self.runner.ctx_size.get() as usize, 1);
        let token_list_len = token_list.len();
        for (i, token) in token_list.into_iter().enumerate() {
            batch.add(token, i as i32, &[0], i == token_list_len - 1)?;
        }
        self.ctx.as_mut().unwrap().decode(&mut batch)?;

        // Generate preparation
        let mut preparation = Runtime {
            sampler: self.req.sampling.to_llama(),
            decoder: UTF_8.new_decoder(),
            batch,
            n_past: token_list_len as i32,
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

impl<'a, M, R> Iterator for Gemma3Stream<'a, M, R>
where
    Self: PrepareRun,
{
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
        let model = self.model;
        let step_copy = step.clone();
        let mut sample = move || -> Result<Option<String>, RunnerError> {
            let token = (&sampler.sample(ctx, -1)).clone();
            sampler.accept(token);
            if model.is_eog_token(token) {
                return Ok(None);
            }
            batch.clear();
            batch.add(token, *n_past + (step_copy as i32), &[0], true)?;

            ctx.decode(batch)?;

            let piece = model.token_to_piece(token, decoder, true, None)?;
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

impl<'a, M, R> Gemma3Stream<'a, M, R> {
    fn new(
        source: Result<LlamaContext<'a>, RunnerError>,
        req: RunnerRequest<M>,
        runner: &'a R,
        model: &'a LlamaModel,
    ) -> Self {
        Self {
            ctx_source: Some(source),
            ctx: None,
            req,
            runner,
            model,
            runtime: None,
            done: false,
        }
    }
}

fn build_hf_api() -> Result<hf_hub::api::tokio::Api, hf_hub::api::tokio::ApiError> {
    let mut api = ApiBuilder::new()
        .with_progress(std::io::stdin().is_terminal())
        .with_token(std::env::var("HF_TOKEN").ok())
        .with_chunk_size(Some(2 << 28));
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        api = api.with_endpoint(endpoint);
    }
    if let Ok(cache) = std::env::var("HF_HOME") {
        api = api.with_cache_dir(
            PathBuf::from_str(&cache).expect("HF_HOME env var is not a valid path"),
        );
    }
    api.build()
}
