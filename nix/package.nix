{
  lib,
  stdenv,
  autoAddDriverRunpath,
  crane,
  pkgs,
  rustPlatform,
  cudaPackages,
  features ? [ ],
  pkg-config,
  openssl,
  cmake,
  version ? "dev",
  ...
}:
let
  cudaSupport = lib.elem "cuda" features;
  effectiveStdenv = if cudaSupport then cudaPackages.backendStdenv else stdenv;

  # Use crane's built-in source cleaning
  promptOrConstraintOrCargo =
    path: type:
    (builtins.match ".*prompt/.*$" path != null)
    || (builtins.match ".*constraint/.*$" path != null)
    || (craneLib.filterCargoSources path type);

  cudaBuildInputs = with cudaPackages; [
    cuda_cccl
    cuda_cudart
    libcublas
  ];

  nativeBuildInputs = [
    pkg-config
    cmake
    rustPlatform.bindgenHook
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [
    openssl
  ]
  ++ lib.optionals cudaSupport cudaBuildInputs;

  env = lib.optionalAttrs cudaSupport {
    CUDA_COMPUTE_CAP = "89";
    RUSTFLAGS = builtins.concatStringsSep " " [
      "-L ${cudaPackages.cuda_cudart}/lib"
      "-L ${cudaPackages.cuda_cudart}/lib/stubs"
      "-L ${cudaPackages.libcublas.lib}/lib"
      "-L ${cudaPackages.libcublas.static}/lib"
    ];
  };

  craneLib = (crane.mkLib pkgs).overrideScope (
    final: prev: {
      stdenvSelector = p: effectiveStdenv;
    }
  );

  # Common args shared between dep-only and full builds
  commonArgs = {
    inherit
      nativeBuildInputs
      buildInputs
      env
      ;

    cargoExtraArgs = lib.concatMapStringsSep " " (f: "--features ${f}") features;
    # Tell crane not to run tests in the build phase
    doCheck = false;
  };

  # Build only dependencies first (allows caching the heavy compile step)
  cargoArtifacts = craneLib.buildDepsOnly (
    commonArgs
    // {
      src = craneLib.cleanCargoSource ../.;
      version = "0.0.0";
    }
  );

in
craneLib.buildPackage (
  commonArgs
  // {
    inherit cargoArtifacts;
    pname = "ledoxide";
    inherit version;
    src = lib.sources.cleanSourceWith {
      src = ../.;
      filter = promptOrConstraintOrCargo;
      name = "source";
    };
    APP_VERSION = version;
  }
)
