locals {
  name = "${var.project}-${var.environment}"
  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

module "kms" {
  source      = "../../modules/kms"
  description = "Phase Runner data key"
  alias       = "${local.name}-data"
  policy      = null
  tags        = local.tags
}

module "vpc" {
  source        = "../../modules/vpc"
  name          = local.name
  cidr_block    = "10.10.0.0/16"
  enable_nat    = true
  public_subnets = {
    a = { cidr = "10.10.0.0/24", az = "${var.region}a" }
    b = { cidr = "10.10.1.0/24", az = "${var.region}b" }
  }
  private_subnets = {
    app = { cidr = "10.10.10.0/24", az = "${var.region}a" }
    data = { cidr = "10.10.11.0/24", az = "${var.region}b" }
  }
  tags = local.tags
}

resource "aws_security_group" "rds" {
  name        = "${local.name}-rds"
  description = "Database access"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "App access"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

module "data_bucket" {
  source            = "../../modules/s3"
  bucket_name       = "${local.name}-data"
  force_destroy     = false
  versioning_enabled = true
  kms_key_arn       = module.kms.key_arn
  tags              = local.tags
}

module "database" {
  source                 = "../../modules/rds"
  name                   = "${local.name}-db"
  subnet_ids             = module.vpc.private_subnet_ids
  vpc_security_group_ids = [aws_security_group.rds.id]
  username               = var.db_username
  password               = var.db_password
  kms_key_id             = module.kms.key_arn
  tags                   = local.tags
}

module "waf" {
  source      = "../../modules/waf"
  name        = "${local.name}-waf"
  description = "Phase Runner edge protection"
  scope       = "CLOUDFRONT"
  tags        = local.tags
}

resource "aws_iam_role" "app" {
  name = "${local.name}-app"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = local.tags
}

resource "aws_iam_policy" "app_data" {
  name        = "${local.name}-app-data"
  description = "Least privilege access to app data"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = ["${module.data_bucket.bucket_arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"],
        Resource = module.kms.key_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "app_data" {
  role       = aws_iam_role.app.name
  policy_arn = aws_iam_policy.app_data.arn
}
