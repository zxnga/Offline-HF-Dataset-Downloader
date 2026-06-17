import shutil
import tarfile
from pathlib import Path


def make_tar_gz(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_arcname = None

    try:
        archive_relative = archive_path.resolve().relative_to(source_dir.resolve())
        archive_arcname = (Path(source_dir.name) / archive_relative).as_posix()
    except ValueError:
        pass

    def exclude_archive(tarinfo: tarfile.TarInfo):
        if archive_arcname is not None and tarinfo.name == archive_arcname:
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name, filter=exclude_archive)


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def remove_output_dir(output_dir: Path) -> None:
    resolved_output_dir = output_dir.resolve()

    if resolved_output_dir.parent == resolved_output_dir:
        raise ValueError(f"Refusing to remove filesystem root: {output_dir}")

    if not output_dir.exists():
        return

    print(f"Removing downloaded dataset folder: {output_dir}")
    shutil.rmtree(output_dir)
