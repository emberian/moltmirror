#!/usr/bin/env python3
"""
Launch EC2 spot workers for distributed scraping.
"""

import boto3
import base64
import time
import json
import sys

BUCKET = "moltbook-archive-319933937176"
INSTANCE_TYPE = "t3.micro"  # ~$0.003/hr spot
AMI_ID = "ami-0c7217cdde317cfec"  # Amazon Linux 2023 (us-east-1) - update for your region

# Get default VPC and subnet
ec2 = boto3.client('ec2')
s3 = boto3.client('s3')

def get_default_vpc_subnet():
    """Get default VPC and a public subnet"""
    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
    if not vpcs['Vpcs']:
        raise Exception("No default VPC found")
    vpc_id = vpcs['Vpcs'][0]['VpcId']

    subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    subnet_id = subnets['Subnets'][0]['SubnetId']
    return vpc_id, subnet_id

def create_security_group(vpc_id):
    """Create security group allowing outbound only"""
    try:
        resp = ec2.create_security_group(
            GroupName='moltbook-scraper',
            Description='Moltbook scraper workers',
            VpcId=vpc_id
        )
        sg_id = resp['GroupId']
        print(f"Created security group: {sg_id}")
    except ec2.exceptions.ClientError as e:
        if 'InvalidGroup.Duplicate' in str(e):
            groups = ec2.describe_security_groups(GroupNames=['moltbook-scraper'])
            sg_id = groups['SecurityGroups'][0]['GroupId']
            print(f"Using existing security group: {sg_id}")
        else:
            raise
    return sg_id

def get_or_create_instance_profile():
    """Get or create IAM role for workers"""
    iam = boto3.client('iam')
    role_name = 'moltbook-scraper-role'
    profile_name = 'moltbook-scraper-profile'

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]
        }]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy)
        )
        print(f"Created IAM role: {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using existing IAM role: {role_name}")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName='s3-access',
        PolicyDocument=json.dumps(s3_policy)
    )

    try:
        iam.create_instance_profile(InstanceProfileName=profile_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name,
            RoleName=role_name
        )
        print(f"Created instance profile: {profile_name}")
        time.sleep(10)  # Wait for propagation
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using existing instance profile: {profile_name}")

    return profile_name

def launch_workers(num_workers, task_type):
    """Launch spot instances as workers"""
    vpc_id, subnet_id = get_default_vpc_subnet()
    sg_id = create_security_group(vpc_id)
    profile_name = get_or_create_instance_profile()

    instance_ids = []

    for worker_id in range(num_workers):
        user_data = f'''#!/bin/bash
set -ex
cd /tmp

# Install Python
yum install -y python3 python3-pip

# Install deps
pip3 install requests boto3

# Download worker
aws s3 cp s3://{BUCKET}/worker.py worker.py

# Run worker
python3 worker.py {worker_id} {num_workers} {task_type}

# Shutdown when done
shutdown -h now
'''

        try:
            resp = ec2.run_instances(
                ImageId=AMI_ID,
                InstanceType=INSTANCE_TYPE,
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet_id,
                SecurityGroupIds=[sg_id],
                IamInstanceProfile={'Name': profile_name},
                InstanceMarketOptions={
                    'MarketType': 'spot',
                    'SpotOptions': {'SpotInstanceType': 'one-time'}
                },
                UserData=base64.b64encode(user_data.encode()).decode(),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'moltbook-worker-{worker_id}'},
                        {'Key': 'Project', 'Value': 'moltbook-archive'}
                    ]
                }]
            )
            instance_id = resp['Instances'][0]['InstanceId']
            instance_ids.append(instance_id)
            print(f"Launched worker {worker_id}: {instance_id}")

        except Exception as e:
            print(f"Failed to launch worker {worker_id}: {e}")

    return instance_ids

def check_status():
    """Check worker completion status"""
    paginator = s3.get_paginator('list_objects_v2')
    done_files = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix='status/'):
        for obj in page.get('Contents', []):
            done_files.append(obj['Key'])
    return done_files

def merge_results(task_type):
    """Merge worker results"""
    paginator = s3.get_paginator('list_objects_v2')
    all_data = []

    prefix = f'{task_type}/'
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                print(f"Downloading {obj['Key']}...")
                resp = s3.get_object(Bucket=BUCKET, Key=obj['Key'])
                data = json.loads(resp['Body'].read())
                all_data.extend(data)

    print(f"Total {task_type}: {len(all_data)}")
    return all_data

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python launch_workers.py launch <num_workers> <task_type>")
        print("  python launch_workers.py status")
        print("  python launch_workers.py merge <task_type>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'launch':
        num_workers = int(sys.argv[2])
        task_type = sys.argv[3]
        print(f"Launching {num_workers} workers for {task_type}...")
        ids = launch_workers(num_workers, task_type)
        print(f"\nLaunched {len(ids)} instances")
        print("Monitor with: python launch_workers.py status")

    elif cmd == 'status':
        done = check_status()
        print(f"Completed workers: {len(done)}")
        for f in done:
            print(f"  {f}")

    elif cmd == 'merge':
        task_type = sys.argv[2]
        data = merge_results(task_type)
        output_file = f'archive/data/{task_type}_merged.json'
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
