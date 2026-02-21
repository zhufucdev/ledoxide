use std::sync::{Arc, LazyLock, Mutex};

use serde::{
    Deserialize, Serialize,
    de::{Unexpected, Visitor},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bill {
    pub notes: String,
    pub amount: f32,
    pub category: Option<Category>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Category(usize);

impl Category {
    pub fn name(&self) -> String {
        CATEGORIES.lock().unwrap().as_ref().unwrap()[self.0].clone()
    }

    pub fn all_cases() -> Vec<Category> {
        Vec::from_iter(
            (0..CATEGORIES.lock().unwrap().as_ref().unwrap().len()).map(|idx| Category(idx)),
        )
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<Category> {
        CATEGORIES
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .iter()
            .position(|n| n == name.as_ref())
            .map(|idx| Category(idx))
    }

    pub fn load_from_names<Iter>(iter: Iter)
    where
        Iter: IntoIterator,
        Iter::Item: AsRef<str>,
    {
        let names = Vec::from_iter(iter.into_iter().map(|name| name.as_ref().to_string()));
        *CATEGORIES.lock().unwrap() = Some(names);
    }
}

static CATEGORIES: LazyLock<Arc<Mutex<Option<Vec<String>>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

impl Serialize for Category {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name().as_str())
    }
}

impl<'de> Deserialize<'de> for Category {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = deserializer.deserialize_string(CategoryNameVisitor)?;
        Ok(
            Category::from_name(&name).ok_or(serde::de::Error::invalid_value(
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
