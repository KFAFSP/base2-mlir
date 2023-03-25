// RUN: base2-opt --lower-funnel-shift %s | FileCheck %s

module {

// CHECK-LABEL: func.func @shl(
// CHECK-SAME: %[[VALUE:.+]]: si64,
// CHECK-SAME: %[[AMOUNT:.+]]: index,
// CHECK-SAME: %[[FUNNEL:.+]]: si64
func.func @shl(%arg0: si64, %arg1: index, %arg2: si64) -> si64 {
    %0 = bit.shl %arg0:%arg2, %arg1 : si64
    // CHECK-DAG: %[[WIDTH:.+]] = index.constant 64
    // CHECK-DAG: %[[INV_AMOUNT:.+]] = index.sub %[[WIDTH]], %[[AMOUNT]]
    // CHECK-DAG: %[[SHL:.+]] = bit.shl %[[VALUE]], %[[AMOUNT]]
    // CHECK-DAG: %[[SHR:.+]] = bit.shr %[[FUNNEL]], %[[INV_AMOUNT]]
    // CHECK-DAG: %[[OR:.+]] = bit.or %[[SHL]], %[[SHR]]
    // CHECK: return %[[OR]]
    return %0 : si64
}

// CHECK-LABEL: func.func @shr(
// CHECK-SAME: %[[VALUE:.+]]: si64,
// CHECK-SAME: %[[AMOUNT:.+]]: index,
// CHECK-SAME: %[[FUNNEL:.+]]: si64
func.func @shr(%arg0: si64, %arg1: index, %arg2: si64) -> si64 {
    %0 = bit.shr %arg2:%arg0, %arg1 : si64
    // CHECK-DAG: %[[WIDTH:.+]] = index.constant 64
    // CHECK-DAG: %[[INV_AMOUNT:.+]] = index.sub %[[WIDTH]], %[[AMOUNT]]
    // CHECK-DAG: %[[SHR:.+]] = bit.shr %[[VALUE]], %[[AMOUNT]]
    // CHECK-DAG: %[[SHL:.+]] = bit.shl %[[FUNNEL]], %[[INV_AMOUNT]]
    // CHECK-DAG: %[[OR:.+]] = bit.or %[[SHR]], %[[SHL]]
    // CHECK: return %[[OR]]
    return %0 : si64
}

}
