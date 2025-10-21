# Volumes Directory

This directory contains all persistent data for the Bahar Docker deployment.

## Structure

```
volumes/
├── transformers_cache/    # HuggingFace transformers models (2-5 GB)
├── huggingface_cache/     # HuggingFace hub cache (1-2 GB)
├── torch_cache/           # PyTorch models (500 MB)
├── nltk_data/             # NLTK datasets (200 MB)
├── spacy_data/            # spaCy models (1 GB)
├── app_data/              # Application data (100 MB)
└── nginx_logs/            # Nginx access and error logs (production only)
```

## Purpose

These directories are mounted as volumes in the Docker containers to persist:
- Downloaded ML models (transformers, spaCy, etc.)
- NLTK data
- Application configuration and data
- Nginx logs (in production mode)

## Benefits of Local Volumes

✅ **Easy Backup**: Just copy the `volumes/` directory
✅ **Easy Migration**: Move to another server by copying files
✅ **Easy Inspection**: Browse files directly on host system
✅ **Easy Cleanup**: Delete specific subdirectories to clear cache
✅ **Version Control**: Can track size and changes

## Disk Space

Total expected size: **5-10 GB** depending on models used

Check current size:
```bash
du -sh volumes/
```

## Backup

Backup all volumes:
```bash
tar czf bahar-volumes-backup-$(date +%Y%m%d).tar.gz volumes/
```

Restore from backup:
```bash
tar xzf bahar-volumes-backup-YYYYMMDD.tar.gz
```

## Cleanup

Remove all cached data (will be re-downloaded on next start):
```bash
rm -rf volumes/*/
```

Remove specific cache:
```bash
rm -rf volumes/transformers_cache/
```

## Permissions

The directories are created automatically by Docker with appropriate permissions.

If you encounter permission issues:
```bash
sudo chown -R $(id -u):$(id -g) volumes/
```

## .gitignore

All contents are ignored by git (except this README and .gitignore) to prevent committing large model files.

