self:
{ pkgs, ... }:
{
  nixpkgs.overlays = [ self.overlay ];
}
