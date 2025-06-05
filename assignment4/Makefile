# FastFlow implementations
FF_PIPELINE = $(SRC_DIR)/fastflow/ff_pipeline_mergesort.cpp
FF_FARM = $(SRC_DIR)/fastflow/ff_farm_mergesort.cpp
FF_ALL2ALL = $(SRC_DIR)/fastflow/ff_all2all_mergesort.cpp

# Single node main
SINGLE_NODE_MAIN = $(SRC_DIR)/main/single_node_main.cpp

# Build single node executable
single_node: $(BUILD_DIR)/mergesort_single

$(BUILD_DIR)/mergesort_single: $(SINGLE_NODE_MAIN) $(COMMON_OBJS) $(SEQ_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJS) $(SEQ_OBJS) \
	$(FF_PIPELINE) $(FF_FARM) $(LDFLAGS)

# Individual FF targets for testing
$(BUILD_DIR)/ff_pipeline_test: $(FF_PIPELINE) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -DTEST_MAIN -o $@ $< $(COMMON_OBJS) $(LDFLAGS)

$(BUILD_DIR)/ff_farm_test: $(FF_FARM) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -DTEST_MAIN -o $@ $< $(COMMON_OBJS) $(LDFLAGS)
