from pathlib import Path
import tomllib

def load_config():
    #__file__ heisst einfach dass halt diese File benutzt wird, also config_loader.py und halt der Pfad also sollte man checken
    base_path = Path(__file__).resolve()

    #3Level höher als der base_path der eben angelegt wurde
    project_root = base_path.parents[3]

    #hier liegt die config
    config_path = project_root / "resources" / "exampleConfigKNN.toml"

    #liefert einfach die toml als bytes, weil die lib das halt so will, anders geht es nicht
    #Also es geht schon aber mit mehr Kopfschmerzen
    with open(config_path, "rb") as f:
        return tomllib.load(f), config_path