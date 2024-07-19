import sys

from teren import scripts

if __name__ == "__main__":
    script_name = sys.argv[1]
    script = getattr(scripts, script_name)
    script.main(sys.argv)
