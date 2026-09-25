{
  pkgs,
  config,
  lib,
  ...
}:
let
  cfg = config.services.ledoxide;
  ledoxide = cfg.package;
in
{

  options.services.ledoxide = {
    enable = lib.mkEnableOption "ledoxide, server to a client-side pulling based VLM bookkeeping workflow";
    package = lib.mkPackageOption pkgs "ledoxide" { };
    bind = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "address:port for the HTTP server to listen on.";
    };
    authKey = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Authorization key gating clients. Disabled if either this or the file variant is unset or set to empty string.";
    };
    authKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File storing the secret authorization key, formatting `AUTH_KEY=[a-zA-Z_-]{31,}`";
    };
    captionModel = lib.mkOption {
      type = lib.types.str;
      default = null;
      description = "Caption model for describing screenshots";
    };
    extractModel = lib.mkOption {
      type = lib.types.str;
      default = null;
      description = "Extract model for amount & category analysis";
    };
    extraEnv = lib.mkOption {
      default = null;
      type = with lib; types.nullOr (types.attrsOf types.str);
      description = "Extra environment variables to use.";
    };
    extraOpts = lib.mkOption {
      default = null;
      type = lib.types.nullOr lib.types.str;
      description = "Extra command line options to use.";
    };
  };

  config = lib.mkIf cfg.enable {
    launchd.user.agents.ledoxide =
      let
        args = lib.concatStringsSep " " [
          (lib.optionalString (cfg.bind != null) "--bind ${cfg.bind}")
          (lib.optionalString (cfg.authKey != null) "--auth-key ${cfg.authKey}")
          (lib.optionalString (cfg.extractModel != null) "--extract-model ${cfg.extractModel}")
          (lib.optionalString (cfg.captionModel != null) "--caption-model ${cfg.captionModel}")
          (lib.optionalString (cfg.extraOpts != null) cfg.extraOpts)
        ];
      in
      {
        path = [ ledoxide ];
        script = ''
          ${lib.getExe pkgs.dotenvy} -f ${cfg.authKeyFile} ${lib.getExe ledoxide} ${args}
        '';
        serviceConfig = {
          RunAtLoad = true;
          KeepAlive = true;
          EnvironmentVariables = lib.optionalAttrs (cfg.extraEnv != null) cfg.extraEnv;
          StandardOutPath = "/tmp/ledoxide.log";
          StandardErrorPath = "/tmp/ledoxide.err.log";
        };

        managedBy = "services.ledoxide.enable";
      };
  };
}
