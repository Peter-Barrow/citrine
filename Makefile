.PHONY: lock
lock:
	nix run .#default.lock

.PHONY: shell
shell:
	nix develop -c /bin/zsh
