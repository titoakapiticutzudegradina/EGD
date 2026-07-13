#!/usr/bin/env python3
from __future__ import annotations

import datetime
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = Path(__file__).resolve().parents[1] / "certs"
DAYS_VALID = 825


def main() -> None:
    #create the cert directory if it doesn t exist
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key_path = CERT_DIR / "server.key"
    cert_path = CERT_DIR / "server.crt"

    #generate key and certificate subject
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "RO"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Licenta"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    #generate subject alternative name
    san = x509.SubjectAlternativeName(
        [
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
    )

    #generate certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=DAYS_VALID)
        )
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Wrote {key_path}")
    print(f"Wrote {cert_path}")
    print("Start server with: USE_TLS=1 python run_server.py")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Install dependencies: pip install cryptography", file=sys.stderr)
        sys.exit(1)
