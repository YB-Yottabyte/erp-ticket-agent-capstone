from pathlib import Path
from erpsim.build_world import World, write, TAXONOMY

def generate_split(split_name, start_seed):
    output_dir = Path("evaluation_cases") / split_name
    output_dir.mkdir(parents=True,exist_ok=True)
    fault_codes = [
        code
        for code in TAXONOMY
        if code != "UNKNOWN"
    ]
    for index, fault_code in enumerate(fault_codes):
        seed = start_seed + index
        world = World(seed)
        world.build_base()
        world.inject(fault_code)
        output_path = (output_dir/ f"{fault_code.lower()}.db")
        write(world,str(output_path))
        print(f"{split_name}: "f"{fault_code} "f"seed={seed}")

def generate_cases():
    generate_split("dev",1000)
    generate_split("test",2000)
if __name__ == "__main__":
    generate_cases()