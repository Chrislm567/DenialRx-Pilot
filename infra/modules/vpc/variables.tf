variable "name" {
  description = "Name prefix for the VPC"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "public_subnets" {
  description = "Map of public subnet definitions"
  type = map(object({
    cidr = string
    az   = string
  }))
}

variable "private_subnets" {
  description = "Map of private subnet definitions"
  type = map(object({
    cidr = string
    az   = string
  }))
}

variable "enable_nat" {
  description = "Whether to create NAT gateways for private subnets"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}
