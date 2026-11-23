# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

obj-m += kretprobe.o

all:
	        make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	        make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
