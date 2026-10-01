use std::sync::{Arc, LazyLock, Mutex};

use serde::{
    Deserialize, Serialize,
    de::{Unexpected, Visitor},
};
use smol_str::{SmolStr, ToSmolStr};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bill {
    pub notes: SmolStr,
    pub amount: f32,
    pub category: Option<SmolStr>,
}

pub trait Category {
    fn name(&self) -> SmolStr;
    fn description(&self) -> Option<SmolStr>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedCategory(usize);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnedCategory {
    name: SmolStr,
    description: Option<SmolStr>,
}

impl SharedCategory {
    fn store(&self) -> OwnedCategory {
        CATEGORIES.lock().unwrap().as_ref().unwrap()[self.0].clone()
    }

    pub fn all_cases() -> Box<[SharedCategory]> {
        Box::from_iter(
            (0..CATEGORIES.lock().unwrap().as_ref().unwrap().len()).map(|idx| SharedCategory(idx)),
        )
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<SharedCategory> {
        CATEGORIES
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .iter()
            .position(|n| n.name == name.as_ref())
            .map(|idx| SharedCategory(idx))
    }

    pub fn load_from_name_desc_pairs<Iter, A, B>(iter: Iter)
    where
        Iter: IntoIterator<Item = (A, Option<B>)>,
        A: AsRef<str>,
        B: AsRef<str>,
    {
        let categories = iter
            .into_iter()
            .map(|(name, description)| OwnedCategory {
                name: name.as_ref().to_smolstr(),
                description: description.map(|it| it.as_ref().to_smolstr()),
            })
            .collect();
        *CATEGORIES.lock().unwrap() = Some(categories);
    }
}

impl OwnedCategory {
    pub fn only_name(name: impl AsRef<str>) -> Self {
        Self {
            name: name.as_ref().to_smolstr(),
            description: None,
        }
    }

    pub fn new(name: impl AsRef<str>, description: impl AsRef<str>) -> Self {
        Self {
            name: name.as_ref().to_smolstr(),
            description: Some(description.as_ref().to_smolstr()),
        }
    }
}

impl Category for SharedCategory {
    fn name(&self) -> SmolStr {
        self.store().name
    }

    fn description(&self) -> Option<SmolStr> {
        self.store().description
    }
}

impl Category for OwnedCategory {
    fn name(&self) -> SmolStr {
        self.name.clone()
    }

    fn description(&self) -> Option<SmolStr> {
        self.description.clone()
    }
}

impl Into<OwnedCategory> for SharedCategory {
    fn into(self) -> OwnedCategory {
        self.store()
    }
}

impl Into<OwnedCategory> for &SharedCategory {
    fn into(self) -> OwnedCategory {
        self.store()
    }
}

static CATEGORIES: LazyLock<Arc<Mutex<Option<Vec<OwnedCategory>>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

impl Serialize for SharedCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name().as_str())
    }
}

impl<'de> Deserialize<'de> for SharedCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = deserializer.deserialize_string(CategoryNameVisitor)?;
        Ok(
            SharedCategory::from_name(&name).ok_or(serde::de::Error::invalid_value(
                Unexpected::Str(&name),
                &CategoryNameVisitor,
            ))?,
        )
    }
}

struct CategoryNameVisitor;
impl<'de> Visitor<'de> for CategoryNameVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a registered category name")
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(v)
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(v.to_string())
    }
}
