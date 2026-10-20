#pragma once
#include <gtest/gtest.h>
#include <acl/acl.h>
#include "mllm/backends/ascend/AscendAllocator.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"

TEST(AscendDeviceOwnershipTest, InitializationKeepsDefaultDevice) {
  int32_t device = -1;
  ASSERT_EQ(aclrtGetDevice(&device), ACL_SUCCESS);
  EXPECT_EQ(device, 0);
}

TEST(AscendDeviceOwnershipTest, FreeUsesOwningPoolAfterDeviceSwitch) {
  uint32_t count = 0;
  ASSERT_EQ(aclrtGetDeviceCount(&count), ACL_SUCCESS);
  if (count < 2) GTEST_SKIP() << "Requires two devices";
  int32_t previous = 0;
  ASSERT_EQ(aclrtGetDevice(&previous), ACL_SUCCESS);
  struct Restore { int32_t id; ~Restore() { aclrtSetDevice(id); } } restore{previous};
  ASSERT_EQ(aclrtSetDevice(1), ACL_SUCCESS);
  mllm::ascend::AscendAllocator allocator;
  mllm::Storage first;
  first.size_ = 4096;
  ASSERT_TRUE(allocator.alloc(&first));
  void* original = first.ptr_;
  ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
  allocator.free(&first);
  EXPECT_EQ(first.ptr_, nullptr);
  int32_t current = -1;
  ASSERT_EQ(aclrtGetDevice(&current), ACL_SUCCESS);
  EXPECT_EQ(current, 0);
  ASSERT_EQ(aclrtSetDevice(1), ACL_SUCCESS);
  mllm::Storage second;
  second.size_ = 4096;
  ASSERT_TRUE(allocator.alloc(&second));
  EXPECT_EQ(second.ptr_, original) << "The block must return to device 1's pool";
  allocator.free(&second);
}
