# GPU Runners Implementation Status

**Last Updated**: 2026-06-17
**Status**: Phase 1 - Code Complete, AWS Wiring Validated, AMI Smoke Test Pending

## ✅ Completed Tasks

### Code Implementation (Ready to Use)

#### 1. GitHub Actions Workflow ✅

**File**: `.github/workflows/record-vllm-gpu-tests.yml`

- ✅ Manual workflow_dispatch trigger
- ✅ OIDC authentication for AWS (no long-lived credentials!)
- ✅ Multi-AZ fallback strategy within us-east-2
- ✅ Hosted cleanup job waits for the GPU job and can terminate the EC2 runner if the self-hosted job never starts
- ✅ Security hardening (`permissions: {}` on test job)
- ✅ Cleanup guarantee (`if: always()`) with empty-output guards
- ✅ Comprehensive error handling and logging

**Features**:

- Runs `gpt-oss:20b` on `g6.2xlarge`
- Select test suite: base, responses, vllm-reasoning

#### 2. Setup vLLM GPU Action ✅

**File**: `.github/actions/setup-vllm-gpu/action.yml`

- ✅ GPU verification (nvidia-smi)
- ✅ CUDA environment detection from the AMI-provided NVIDIA/CUDA installation
- ✅ Python virtual environment setup
- ✅ PyTorch with CUDA installation
- ✅ vLLM installation with GPU support in `/tmp/vllm-env`
- ✅ Model pulling (Ollama and HuggingFace formats)
- ✅ vLLM server startup with optimal settings
- ✅ Health check with 10-minute timeout
- ✅ GPT-OSS MXFP4 weights without an extra vLLM quantization flag

#### 3. Launch GPU Runner Action ✅

**File**: `.github/actions/launch-gpu-runner/action.yml`

- ✅ Wrapper for machulav/ec2-github-runner
- ✅ Support for both start and stop modes
- ✅ Configurable instance type, region, subnet, AMI, security group
- ✅ Resource tagging support
- ✅ IAM role support for enhanced security

**Note**: Multi-region fallback logic is handled in the workflow, not the action.

#### 4. Test Configuration ✅

**Files**: `tests/integration/suites.py`, `tests/integration/ci_matrix.json`

- ✅ Added `vllm-gpu-gpt-oss` setup in suites.py
- ✅ Added `gpu-vllm` matrix in ci_matrix.json
- ✅ Configured for base, responses, and vllm-reasoning test suites

#### 5. Documentation ✅

**Files**: `docs/gpu-runners.md`, `AWS_SETUP_GUIDE.md`, `IMPLEMENTATION_PLAN.md`

- ✅ User guide for triggering GPU workflows
- ✅ Step-by-step AWS setup guide with OIDC
- ✅ Detailed implementation plan
- ✅ Troubleshooting guides
- ✅ Cost estimates and monitoring guidance

## 🔧 AWS Setup Status

**Priority**: HIGH - Required before future manual runs

### Infrastructure Tasks

#### 1. Set up OIDC Provider ✅

**Owner**: DevOps
**Time**: 30 minutes

- [x] Create OIDC provider in AWS IAM
- [x] Create IAM role scoped to `ogx-ai/ogx`
- [x] Configure trust policy for GitHub
- [x] Attach EC2 permissions policy
- [x] Save role ARN for GitHub secrets

**Guide**: `AWS_SETUP_GUIDE.md` Step 1

#### 2. Set up VPC and Networking ✅

**Owner**: DevOps
**Time**: 1-2 hours

**us-east-2 (Primary)**:

- [x] Create or identify VPC
- [x] Create 3 subnets (us-east-2a, 2b, 2c)
- [x] Configure internet gateway and routing
- [x] Create security group

**Guide**: `AWS_SETUP_GUIDE.md` Step 2

#### 3. Create GPU-Enabled AMI ✅

**Owner**: DevOps
**Time**: 3-4 hours (includes building time)

**us-east-2**:

- [x] Launch g6.2xlarge instance
- [x] Install NVIDIA drivers
- [x] Install CUDA 13.0
- [x] Install/validate vLLM 0.22.1 in the AMI build
- [x] Install Docker with NVIDIA Container Toolkit
- [x] Install Python 3.12
- [x] Create AMI: `ami-090a815de2a7461f2`

The workflow still installs the pinned vLLM version into `/tmp/vllm-env` during
each run so the test environment is reproducible even when the AMI already has
vLLM available.

**Guide**: `AWS_SETUP_GUIDE.md` Step 3

#### 4. Create GitHub Personal Access Token ⏳

**Owner**: Charles
**Time**: 5 minutes

- [ ] Create PAT with `repo` scope
- [ ] Save token securely

**Guide**: `AWS_SETUP_GUIDE.md` Step 4

#### 5. Configure GitHub Secrets and Variables ✅

**Owner**: Charles
**Time**: 10 minutes

**Secrets** (Settings > Secrets and variables > Actions > Secrets):

- [ ] `AWS_ROLE_ARN`: IAM role ARN from step 1
- [ ] `RELEASE_PAT`: GitHub PAT from step 4

**Variables** (Settings > Secrets and variables > Actions > Variables):

- [x] `SUBNET_US_EAST_2A`: `subnet-02d230cffd9385bd4`
- [x] `SUBNET_US_EAST_2B`: `subnet-024298cefa3bedd61`
- [x] `SUBNET_US_EAST_2C`: `subnet-04701a08396b2ed01`
- [x] `AWS_EC2_AMI_US_EAST_2`: `ami-090a815de2a7461f2`
- [x] `SECURITY_GROUP_ID_US_EAST_2`: `sg-06300447c4a5fbef3`

**Guide**: `AWS_SETUP_GUIDE.md` Step 5

## 🧪 Validation Status

**Priority**: MEDIUM - AMI smoke test remains after merge

### Test Plan

#### 1. Initial Test Run ✅

**Owner**: Charles / skamenan
**Time**: 30 minutes

- [x] Trigger workflow via workflow_dispatch
- [x] Verify EC2 launches successfully
- [x] Verify runner registers
- [x] Verify GPU is detected
- [x] Verify vLLM starts
- [x] Verify tests run
- [x] Verify recordings uploaded
- [x] Verify EC2 cleanup

Validated suites:

- [x] `base`
- [x] `responses`
- [x] `vllm-reasoning`

**Guide**: `AWS_SETUP_GUIDE.md` Step 6

#### 2. Failure Scenario Testing ⏳

**Owner**: Charles
**Time**: 1-2 hours

- [ ] Test manual cancellation (verify cleanup)
- [ ] Test job failure (verify cleanup)
- [ ] Verify alternate us-east-2 subnet/AZ fallback (if capacity issue)
- [ ] Run the post-merge AMI smoke test and mark `gpt-oss:20b` end-to-end validation complete

#### 3. Performance Validation ⏳

**Owner**: Charles
**Time**: Ongoing (10+ runs)

- [ ] Measure average execution time (target: < 30 min)
- [ ] Measure success rate (target: > 95%)
- [ ] Verify no orphaned instances
- [ ] Track AWS costs (target: ~$0.43 per run)

## 📋 Phase 2: Optimization (Future)

**Priority**: MEDIUM - Cost and performance improvements

### Tasks

#### 1. Set up AWS Cost Monitoring ⏳

**Time**: 3-4 hours

- [ ] Create AWS Budget with $50/month alert
- [ ] Create CloudWatch alarm for long-running instances
- [ ] Create Lambda for auto-cleanup
- [ ] Enable cost allocation tags
- [ ] Create CloudWatch dashboard

**ROI**: Prevents cost overruns, better visibility

#### 2. Implement Spot Instances ⏳

**Time**: 4-6 hours

- [ ] Update launch-gpu-runner action for spot support
- [ ] Add spot/on-demand fallback logic
- [ ] Test spot reliability (10+ runs)
- [ ] Update documentation

**ROI**: 70-80% cost reduction ($0.43 → $0.09-$0.17 per run)

#### 3. Add Model Caching ⏳

**Time**: 2-3 hours

- [ ] Pre-cache gpt-oss:20b in AMI
- [ ] Test with cached model
- [ ] Measure time savings
- [ ] Update documentation

**ROI**: 33% faster runs (30 min → 20 min)

## 📊 Success Metrics

### Functional

- [ ] Successfully record gpt-oss:20b tests within 30 minutes
- [ ] 95%+ success rate for runner provisioning
- [ ] Zero leaked AWS credentials
- [ ] Zero orphaned EC2 instances

### Performance

- [ ] Runner startup time < 5 minutes
- [ ] vLLM startup time < 5 minutes
- [ ] Total execution time < 30 minutes

### Cost

- [ ] Monthly AWS costs < $20 for on-demand usage
- [ ] Average cost per run: $0.40-$0.50

## 🚀 Quick Start Guide

### For First-Time Setup

1. **AWS Setup** (DevOps + Charles, ~8 hours total):

   ```bash
   # Follow AWS_SETUP_GUIDE.md steps 1-5
   # Estimated time breakdown:
   # - OIDC setup: 30 min
   # - VPC/networking: 1-2 hours
   # - AMI creation: 3-4 hours
   # - GitHub config: 15 min
   # - Testing: 30 min
   ```

2. **Test Workflow**:
   - Go to Actions > vLLM GPU Recording
   - Click Run workflow
   - Use defaults
   - Verify success

3. **Monitor**:
   - Check AWS EC2 console
   - Verify instance cleanup
   - Check costs in AWS Billing

### For Regular Use

1. Go to Actions > vLLM GPU Recording
2. Click Run workflow
3. Select model and test suite
4. Download recordings artifact when done
5. Commit recordings to PR

## 📁 File Structure

```text
llama-stack/
├── .github/
│   ├── actions/
│   │   ├── launch-gpu-runner/
│   │   │   └── action.yml ✅
│   │   └── setup-vllm-gpu/
│   │       └── action.yml ✅
│   └── workflows/
│       └── record-vllm-gpu-tests.yml ✅
├── docs/
│   └── gpu-runners.md ✅
├── tests/integration/
│   ├── ci_matrix.json ✅ (updated)
│   └── suites.py ✅ (updated)
├── AWS_SETUP_GUIDE.md ✅
├── IMPLEMENTATION_PLAN.md ✅
└── IMPLEMENTATION_STATUS.md ✅ (this file)
```

## 🔒 Security Highlights

### OIDC Authentication

- ✅ No long-lived AWS credentials in GitHub
- ✅ Temporary tokens from AWS STS
- ✅ Automatic rotation
- ✅ Better audit trail

### Test Job Isolation

- ✅ `permissions: {}` on test job
- ✅ Cannot access secrets
- ✅ Cannot write to repository
- ✅ Prevents credential theft

### Cleanup Guarantees

- ✅ `if: always()` on cleanup job
- ✅ Runs even on failure/cancellation
- ✅ Prevents orphaned instances
- ✅ Cost protection

### Resource Tagging

- ✅ All instances tagged with:
  - Project, Purpose, Model
  - GitHub repository, run ID
  - ManagedBy: GitHub-Actions

## 💰 Cost Estimates

### Current State (On-Demand)

| Scenario | Frequency | Cost/Month |
|----------|-----------|------------|
| Weekly re-recording | 4x/month | **$1.72** |
| Daily testing | 30x/month | **$12.90** |
| On-demand (PRs) | 10x/month | **$4.30** |

### Phase 2 (With Spot Instances)

| Scenario | Frequency | Cost/Month |
|----------|-----------|------------|
| Weekly re-recording | 4x/month | **$0.36-$0.68** |
| Daily testing | 30x/month | **$2.70-$5.10** |
| On-demand (PRs) | 10x/month | **$0.90-$1.70** |

**Savings**: 60-90% with spot instances

## 📞 Support

### For AWS Setup Issues

- Consult `AWS_SETUP_GUIDE.md`
- Check AWS CloudTrail for API errors
- Verify IAM permissions

### For Workflow Issues

- Consult `docs/gpu-runners.md`
- Check GitHub Actions logs
- Verify all secrets/variables set correctly

### For General Questions

- Review `docs/gpu-runners.md` for architecture
- Review `IMPLEMENTATION_PLAN.md` for roadmap
- Contact: Charles Doern (@cdoern)

## 🎯 Next Actions

**Immediate** (This Week):

1. [ ] Complete AWS infrastructure setup (Tasks 11, 10 from plan)
2. [ ] Configure GitHub secrets and variables
3. [ ] Run first test workflow (Task 8)

**Short-term** (Next 2 Weeks):
4. [ ] Iterate on any issues from testing
5. [ ] Run 10+ workflows to validate reliability
6. [ ] Measure and document actual costs

**Medium-term** (Next Month):
7. [ ] Implement cost monitoring (Task 4)
8. [ ] Implement spot instances (Task 3)
9. [ ] Add model caching (Task 5)

---

**Status**: Ready for AWS setup and testing! All code is complete and documented. 🚀
