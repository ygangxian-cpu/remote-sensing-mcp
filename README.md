# Remote Sensing MCP

Online remote-sensing MCP gateway for the LST downscaling workflow.

Deployment architecture:
- Vercel Hobby: lightweight public MCP gateway.
- GitHub Actions: heavy ELITE FY-4A monthly ZIP/HDF processing.
- Google Earth Engine: optional authenticated discovery/export after service-account configuration.

Public tools that work without secrets:
- service_status
- list_supported_datasets
- elite_fy4a_lst_catalog
- plan_elite_fy4a_lst_download

Optional Earth Engine tools require Vercel environment variables:
- EE_PROJECT
- EE_SERVICE_ACCOUNT_JSON or EE_SERVICE_ACCOUNT_JSON_BASE64
- REMOTE_MCP_TOKEN
- GEE_GCS_BUCKET for export_era5_land_to_gcs

Optional remote ELITE job submission requires:
- GITHUB_WORKFLOW_TOKEN
- REMOTE_MCP_TOKEN
- GITHUB_WORKFLOW_REPOSITORY=yuyan3616/first-repo
- GITHUB_WORKFLOW_ID=remote-sensing-elite.yml
- GITHUB_WORKFLOW_REF=main

Endpoints after deployment:
- /health
- /mcp

Security rule:
If any privileged credential is configured, REMOTE_MCP_TOKEN is mandatory and MCP requests must send:
Authorization: Bearer <REMOTE_MCP_TOKEN>

The ELITE worker downloads only the required monthly Zenodo archives, applies scale 0.01, geolocates the FY-4A/AGRI 4 km full disk, crops to the requested WGS84 bbox, writes EPSG:4326 GeoTIFFs, and uploads a short-retention GitHub Actions artifact.
