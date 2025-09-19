terraform {
  required_version = ">= 1.5.0"
}

resource "aws_db_subnet_group" "this" {
  name       = "${var.name}-subnet-group"
  subnet_ids = var.subnet_ids
  tags       = merge(var.tags, { Name = "${var.name}-subnet-group" })
}

resource "aws_db_instance" "this" {
  identifier              = var.name
  engine                  = var.engine
  engine_version          = var.engine_version
  instance_class          = var.instance_class
  allocated_storage       = var.allocated_storage
  max_allocated_storage   = var.max_allocated_storage
  db_subnet_group_name    = aws_db_subnet_group.this.id
  vpc_security_group_ids  = var.vpc_security_group_ids
  username                = var.username
  password                = var.password
  skip_final_snapshot     = var.skip_final_snapshot
  backup_retention_period = var.backup_retention_period
  multi_az                = var.multi_az
  storage_encrypted       = true
  kms_key_id              = var.kms_key_id
  publicly_accessible     = false
  deletion_protection     = true
  apply_immediately       = true
  performance_insights_enabled = true
  monitoring_interval     = 60

  tags = merge(var.tags, { Name = var.name })
}

output "endpoint" {
  value = aws_db_instance.this.endpoint
}

output "arn" {
  value = aws_db_instance.this.arn
}
