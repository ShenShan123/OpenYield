"""Generate a compact complete macro with reusable compiler subcircuits."""
from .scaled_macro import main as scaled_main


def main(argv=None):
    return scaled_main(argv, default_floorplan='compact')


if __name__ == '__main__':
    raise SystemExit(main())
