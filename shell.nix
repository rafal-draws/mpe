{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
	buildInputs = [
		pkgs.python312
		pkgs.python312Packages.pip
		pkgs.python312Packages.numpy
		pkgs.python312Packages.matplotlib
	];

	shellHook = ''
	echo "env is ready"
	'';
}
