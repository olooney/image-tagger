set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]

python := if os() == "windows" { "./.venv/Scripts/python.exe" } else { "./.venv/bin/python" }

default:
    just --list

convert:
    {{python}} src/cli.py convert

tag:
    {{python}} src/cli.py tag

rename:
    {{python}} src/cli.py rename

scramble:
    {{python}} src/cli.py scramble

gallery:
    {{python}} src/cli.py gallery

clean:
    {{python}} src/cli.py clean

run: convert tag rename gallery

test: clean scramble run
