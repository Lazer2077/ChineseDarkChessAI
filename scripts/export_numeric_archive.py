"""Publish identical numeric payloads with deterministic, empty owner metadata."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--deny-token', action='append', default=[])
    args = p.parse_args()
    source, target, report = map(Path, [args.source, args.target, args.report])
    if target.exists() or report.exists():
        raise ValueError('Refuse to overwrite public archive exports')
    payloads = {}
    forbidden = [b'/' + b'home' + b'/'] + [s.lower().encode() for s in args.deny_token]
    with tarfile.open(source, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or len(name.parts) != 1 or member.name in payloads:
                raise ValueError('Require unique flat numeric archive members')
            data = archive.extractfile(member).read()
            if member.name.endswith('.npz'):
                with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                    for key in arrays.files:
                        a = arrays[key]
                        if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
                            raise ValueError('Require finite numeric arrays')
            elif member.name.endswith('.json'):
                json.loads(data)
                if any(token in data.lower() for token in forbidden):
                    raise ValueError('Local identifiers in numeric archive sidecar')
            else:
                raise ValueError('Unexpected archive member type')
            if any(token in member.name.lower().encode() for token in forbidden):
                raise ValueError('Local identifiers in archive member name')
            payloads[member.name] = data
    with target.open('xb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                for name, data in sorted(payloads.items()):
                    member = tarfile.TarInfo(name)
                    member.size, member.mode, member.mtime = len(data), 0o644, 0
                    archive.addfile(member, io.BytesIO(data))
    with tarfile.open(target, 'r:gz') as archive:
        actual = {}
        for member in archive:
            if (member.uid != 0 or member.gid != 0 or member.uname or member.gname or
                    member.mtime != 0 or member.pax_headers or not member.isfile()):
                raise ValueError('Unexpected public owner/time metadata')
            actual[member.name] = archive.extractfile(member).read()
    if actual != payloads:
        raise ValueError('Public export changed member bytes')
    result = {'schema': 1, 'status': 'completed', 'source_archive_sha256': digest(source.read_bytes()),
        'public_archive': {'path': str(target), 'sha256': digest(target.read_bytes())},
        'members': [{'name': name, 'sha256': digest(data), 'bytes': len(data)} for name, data in sorted(payloads.items())],
        'member_payloads_identical': True,
        'scope': 'Publication packaging only: preserve all numeric arrays/sidecars byte for byte; normalize owner/group/time headers. Frozen experiment runtime digests remain unchanged.'}
    report.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
