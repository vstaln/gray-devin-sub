//! Library target for `gray-devin-sub`: the protocol-1.2 sidecar plugin
//! (`devin-sub` binary) for Gray's plugin system.

pub mod catalog;
pub mod chat;
pub mod manifest;
pub mod models;
pub mod relay;
pub mod setup;

#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
