{
  description = "A Nix-flake-based Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, ... }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      lib = inputs.nixpkgs.lib;
      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [
                inputs.self.overlays.default
              ];
              config.allowUnfree = true;
              config.cudaSupport = true;
            };
          }
        );
    in
    {
      overlays.default = final: prev: {
        rustToolchain =
          with inputs.fenix.packages.${prev.stdenv.hostPlatform.system};
          combine (
            with stable;
            [
              clippy
              rustc
              cargo
              rustfmt
              rust-src
            ]
          );
      };

      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustToolchain
                openssl
                pkg-config
                cargo-deny
                cargo-edit
                cargo-watch
                rust-analyzer

              ]
              ++ lib.optionals stdenv.isLinux [
                cudaPackages.cudatoolkit
                cudaPackages.cuda_nvcc
                cudaPackages.cuda_cudart
              ];

            env = {
              # Required by rust-analyzer
              RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
              CUDA_COMPUTE_CAP = "89";
              CUDATKDIR = "${pkgs.cudaPackages.cudatoolkit}";
              LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudatoolkit}/lib/stubs";
            };
          };
        }
      );
    };
}
