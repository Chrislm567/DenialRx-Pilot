# Phase Runner Infrastructure

This directory defines the Terraform baseline for the Phase Runner stack. The configuration is organized into reusable modules and environment-specific instantiations.

## Structure

- `modules/`
  - `vpc`: Opinionated VPC with public/private subnets and optional NAT gateways.
  - `rds`: Encrypted PostgreSQL RDS instance with monitoring defaults.
  - `s3`: Secure S3 bucket with KMS encryption and bucket keys enabled.
  - `kms`: Customer-managed KMS key with automatic rotation.
  - `waf`: AWS WAFv2 web ACL with managed rule sets.
- `environments/dev`: Example instantiation of the modules.

## Usage

```bash
cd infra/environments/dev
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
```

The CI workflow runs `terraform plan` in plan-only mode for pull requests.

## Variables

See `variables.tf` within each module and environment for configuration options. Secrets such as database passwords must be supplied via Terraform variables or remote state, never committed to the repository.
