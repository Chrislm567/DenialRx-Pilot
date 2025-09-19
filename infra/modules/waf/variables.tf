variable "name" {
  description = "Name of the web ACL"
  type        = string
}

variable "description" {
  description = "Description of the ACL"
  type        = string
  default     = "Phase Runner web ACL"
}

variable "scope" {
  description = "Scope of the ACL"
  type        = string
  default     = "CLOUDFRONT"
}

variable "tags" {
  description = "Tags applied to the ACL"
  type        = map(string)
  default     = {}
}
