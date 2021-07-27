
from secrets import token_hex

N_BYTES: int = 32

if __name__ == "__main__":

    with open(".env", mode="r") as f:
        data = f.read()

    if data:
        lines = data.splitlines()

        for i, line in enumerate(lines):
            if line.startswith("SECRET_KEY"):
                lines[i] = f"SECRET_KEY={token_hex(N_BYTES)}"
                break

        with open(".env", mode="w") as outf:
            outf.write("\n".join(lines))

