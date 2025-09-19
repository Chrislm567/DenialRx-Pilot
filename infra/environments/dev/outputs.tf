output "vpc_id" {
  value = module.vpc.vpc_id
}

output "private_subnet_ids" {
  value = module.vpc.private_subnet_ids
}

output "data_bucket" {
  value = module.data_bucket.bucket_id
}

output "database_endpoint" {
  value       = module.database.endpoint
  description = "Database endpoint"
}

output "waf_arn" {
  value = module.waf.arn
}
