from pathlib import Path
from erpsim.build_world import World, write, TAXONOMY


def generate_cases():
    output_dir = Path("evaluation_cases")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fault_codes = [
        code
        for code in TAXONOMY
        if code != "UNKNOWN"
    ]

    for index, fault_code in enumerate(fault_codes):
        seed = 1000 + index

        world = World(seed)
        world.build_base()
        world.inject(fault_code)

        output_path = (
            output_dir
            / f"{fault_code.lower()}.db"
        )

        write(
            world,
            str(output_path),
        )

        print(
            f"Generated {fault_code} "
            f"with seed {seed}"
        )


if __name__ == "__main__":
    generate_cases()