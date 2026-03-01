#!/usr/bin/env perl

use strict;
use warnings;

my $toml_file      = 'Cargo.toml';
my $lock_file      = 'Cargo.lock';
my $toml_edit_file = 'Cargo.edit';
my $lock_edit_file = 'Cargo.edit.lock';

# --- Cargo.toml: replace version in first 3 lines, write to Cargo.edit ---
open my $toml_in,  '<', $toml_file      or die "Cannot open $toml_file: $!";
open my $toml_out, '>', $toml_edit_file or die "Cannot open $toml_edit_file: $!";

my $line_num = 0;
while (<$toml_in>) {
    $line_num++;
    s/version = "[0-9.]+"/ version = "0.1.0"/ if $line_num <= 3;
    print $toml_out $_;
}

close $toml_in;
close $toml_out;

# --- Cargo.lock: multiline replace of ledoxide version, write to Cargo.edit.lock ---
open my $lock_in,  '<', $lock_file      or die "Cannot open $lock_file: $!";
my $lock_content = do { local $/; <$lock_in> };
close $lock_in;

$lock_content =~ s/(\[\[package\]\]\nname = "ledoxide"\nversion = )"[0-9.]+"/$1"0.1.0"/g;

open my $lock_out, '>', $lock_edit_file or die "Cannot open $lock_edit_file: $!";
print $lock_out $lock_content;
close $lock_out;
