#!/usr/bin/env bash
cat Cargo.toml | head -n 3 | awk '{gsub(/version = "[0-9.]+"/, "version = \"0.1.0\""); print}' > Cargo.edit
cat Cargo.toml | tail -n +4 >> Cargo.edit
