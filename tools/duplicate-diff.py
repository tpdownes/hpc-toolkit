# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import filecmp
import sys

duplicates = [
    [
        "modules/file-system/filestore/scripts/mount.sh",
        "modules/file-system/pre-existing-network-storage/scripts/mount.sh",
    ],
    [
        "community/modules/file-system/nfs-server/scripts/install-nfs-client.sh",
        "modules/file-system/filestore/scripts/install-nfs-client.sh",
        "modules/file-system/pre-existing-network-storage/scripts/install-nfs-client.sh",
    ]
]

for group in duplicates:
    first = group[0]
    for second in group[1:]:
        if not filecmp.cmp(first, second):  # true if files are the same
            print(f'found diff between {first} and {second}')
            sys.exit(1)        
