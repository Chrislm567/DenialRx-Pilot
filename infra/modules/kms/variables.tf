variable "description" {
  description = "Description of the key"
  type        = string
}

variable "alias" {
  description = "Alias name without prefix"
  type        = string
}

variable "policy" {
  description = "IAM policy for the key"
  type        = string
  default     = null
}

variable "tags" {
  description = "Key tags"
  type        = map(string)
  default     = {}
}
