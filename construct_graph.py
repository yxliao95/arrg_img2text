import bisect

from datasets import load_from_disk
from tqdm import tqdm


class Entity:
    def __init__(self, start, end, label, sent_id, tok_list=None, tok_str=None):
        self.id = None
        self.tok_indices = [start, end]
        self.label = label

        self.sent_id = sent_id
        if tok_list:
            self.tok_list = tok_list
            self.tok_str = " ".join(tok_list) if not tok_str else tok_str
        elif tok_str:
            self.tok_str = tok_str
            self.tok_list = tok_str.split(" ")

        if "Observation" in label:
            self.label_type = "OBS"
        elif "Anatomy" == label:
            self.label_type = "ANAT"
        else:
            self.label_type = "LOCATT"

        self.attr_normal = "NA"
        self.attr_action = "NA"
        self.attr_change = "NA"

        self.chain_info = {
            "modify": {"from": [], "to": []},
            "part_of": {"from": [], "to": []},
            "located_at": {"from": [], "to": []},
            "suggestive_of": {"from": [], "to": []},
        }

    def __repr__(self) -> str:
        # return f"{self.tok_str} {self.tok_indices}: {self.label}, {self.attr_normal, self.attr_action, self.attr_change}"
        return f"{self.tok_str}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.tok_indices == other.tok_indices
        else:
            return other == self.tok_indices

    def __hash__(self):
        return hash(str(self.tok_indices))


class Relation:
    def __init__(self, subj_ent, obj_ent, label):
        self.label = label
        self.subj_ent = subj_ent
        self.obj_ent = obj_ent

    def __repr__(self) -> str:
        return f"{self.subj_ent.tok_str} {self.label} {self.obj_ent.tok_str}"

    def __str__(self) -> str:
        return self.__repr__()


class LinkedGraph:
    def __init__(self, ents):
        self.id = None
        self.ents = sorted(ents, key=lambda x: x.tok_indices[0])
        self.rels = []
        self.sent_id = ents[0].sent_id

        assert len(set([i.sent_id for i in ents])) == 1

    def get_involved_rels(self, rel_list):
        target_rels = []
        in_used_ents = set()
        for rel in rel_list:
            if rel.subj_ent in self.ents and rel.obj_ent in self.ents:
                target_rels.append(rel)
                in_used_ents.update([rel.subj_ent, rel.obj_ent])
        self.rels = target_rels

    def __repr__(self) -> str:
        return f"{[i.tok_str for i in self.ents]}"

    def __str__(self) -> str:
        return self.__repr__()


class SentenceRepresentation:
    def __init__(self, doc_key, sent_id, sent_text):
        self.doc_key = doc_key
        self.sent_id = sent_id
        self.sent_text = sent_text
        self.ent_tuples = []
        self.rel_tuples = []

    def set_sent_repr(self, linked_graphs):
        for linked_graph in linked_graphs:
            for ent in linked_graph.ents:
                self.ent_tuples.append((ent.tok_str, ent.label, ent.attr_normal, ent.attr_action, ent.attr_change))

            for rel in linked_graph.rels:
                self.rel_tuples.append((rel.subj_ent.tok_str, rel.label, rel.obj_ent.tok_str))

    def __repr__(self) -> str:
        return f"{self.sent_text}"

    def __str__(self) -> str:
        return self.__repr__()


def search_linked_ents(curr_ent, visited, group):
    visited.add(curr_ent)
    group.append(curr_ent)
    neighbors = [ent for nested_dict in curr_ent.chain_info.values() for adjacent_ents in nested_dict.values() for ent in adjacent_ents]
    for next_ent in neighbors:
        if next_ent not in visited:
            search_linked_ents(next_ent, visited, group)


def max_coverage_spans(spans):
    if not spans:
        return [], 0

    # 按结束时间升序排序
    sorted_spans = sorted(spans, key=lambda x: x[1])
    n = len(sorted_spans)
    starts = [s[0] for s in sorted_spans]
    ends = [s[1] for s in sorted_spans]
    lengths = [e - s for s, e in sorted_spans]

    # 预处理j_values数组，记录每个i对应的最大的j，使得 ends[j] <= starts[i]
    j_values = []
    for i in range(n):
        start_i = starts[i]
        j = bisect.bisect_right(ends, start_i) - 1  # 二分查找, 找到第一个`大于`start_i的位置
        j_values.append(j)

    # 构建 dp 数组，其中 dp[i] 表示前 i+1 个 span 的最大总覆盖率。通过比较包含当前 span 和不包含当前 span 的情况，确定最优解。
    # dp记录了选中下一个span之后的总覆盖率
    dp = [0] * n
    dp[0] = lengths[0]
    for i in range(1, n):
        j = j_values[i]
        current = lengths[i] + (dp[j] if j >= 0 else 0)
        dp[i] = max(dp[i - 1], current)

    # 回溯找出选中的span。从最后一个span开始，如果当前span被选中，则跳到j_values[i]对应的span
    # 当dp发生变化时，说明
    selected_indices = []
    i = n - 1
    while i >= 0:
        if i == 0:
            if dp[i] == lengths[i]:
                selected_indices.append(i)
            break
        if dp[i] > dp[i - 1]:
            selected_indices.append(i)
            i = j_values[i]
        else:
            i -= 1

    selected_indices.reverse()
    selected_spans = [sorted_spans[i] for i in selected_indices]
    total_coverage = dp[-1]

    return selected_indices, selected_spans, total_coverage


def resolve_ent_rel(split_sent_idx, cxrgraph_ent_lst, cxrgraph_rel_lst, cxrgraph_attr_lst, radlex_lst):
    ent_list = []
    rel_list = []
    for ent in cxrgraph_ent_lst:
        ent = Entity(start=ent["tok_indices"][0], end=ent["tok_indices"][1], label=ent["ent_type"], tok_list=ent["ent_toks"], sent_id=split_sent_idx)
        ent_list.append(ent)
    for attr in cxrgraph_attr_lst:
        ent = ent_list[ent_list.index(attr["tok_indices"])]
        ent.attr_normal = attr["attr_normality"]
        ent.attr_action = attr["attr_action"]
        ent.attr_change = attr["attr_change"]
    for rel in cxrgraph_rel_lst:
        subj_ent = ent_list[ent_list.index(rel["subj_tok_indices"])]
        obj_ent = ent_list[ent_list.index(rel["obj_tok_indices"])]
        label = rel["rel_type"]
        subj_ent.chain_info[label]["to"].append(obj_ent)
        obj_ent.chain_info[label]["from"].append(subj_ent)
        rel_list.append(Relation(subj_ent, obj_ent, label))

    # Set ent id
    for ent_idx, ent in enumerate(sorted(ent_list, key=lambda x: x.tok_indices[0])):
        ent.id = f"E{ent_idx}"

    # 选择覆盖率最大的radlex子集
    radlex_ent_indices = [node["tok_indices"] for node in radlex_lst]
    selected_idx_list, _, _ = max_coverage_spans(radlex_ent_indices)

    # 用radlex的ent替换cxrgraph的ent
    for radlex_idx in selected_idx_list:
        radlex_ent = radlex_lst[radlex_idx]
        merged_cxrgraph_ents = []
        for cxrgraph_ent in ent_list:
            # 如果cxrgrpah被radlex包含，那么就加入候选集等待替换；如果cxrgraph和radlex有交集，那么就跳过这个radlex
            pos_ab = check_span_relation(cxrgraph_ent.tok_indices, radlex_ent["tok_indices"])
            if pos_ab in ["equal", "inside"]:
                merged_cxrgraph_ents.append(cxrgraph_ent)
            elif pos_ab == "overlap":
                break
            else:
                continue

        # 如果merged_cxrgraph_ents不为空，那么就用radlex替换候选集的cxrgraph ent
        if merged_cxrgraph_ents:
            inherited_label = get_label_inheritance(merged_cxrgraph_ents)
            inherited_attr_dict = get_attr_inheritance(merged_cxrgraph_ents)

            new_ent = Entity(start=radlex_ent["tok_indices"][0], end=radlex_ent["tok_indices"][1], label=inherited_label, tok_str=radlex_ent["radlex_name"], sent_id=split_sent_idx)
            new_ent.attr_normal = inherited_attr_dict["normality"]
            new_ent.attr_action = inherited_attr_dict["action"]
            new_ent.attr_change = inherited_attr_dict["change"]
            new_ent.id = radlex_ent["radlex_id"]

            # inherit chain info
            for cxrgraph_ent in merged_cxrgraph_ents:
                for rel_type, from_to_dict in cxrgraph_ent.chain_info.items():
                    # 把merged_cxrgraph_ents的from和to的关系都继承过来，如果是内部ents之间指向关系，那么就跳过
                    for key, value_lst in from_to_dict.items():
                        for value in value_lst:
                            if value not in merged_cxrgraph_ents:
                                new_ent.chain_info[rel_type][key].append(value)

            # replace from ent_list
            ent_list.append(new_ent)
            for cxrgraph_ent in merged_cxrgraph_ents:
                ent_list.remove(cxrgraph_ent)

            # replace from rel_list
            # pleural_effusion 应该把 pleural 和 effusion 都替换掉。在rel中则包括：
            #   opacifications suggestive_of effusions
            #   bilateral modify pleural
            #   effusions located_at pleural
            rel_objs_tobe_removed = []
            for rel in rel_list:
                if rel.subj_ent in merged_cxrgraph_ents and rel.obj_ent in merged_cxrgraph_ents:
                    # 关于 from 和 to 的关系链，在新的ent中已经继承了，所以这里不需要处理
                    rel_objs_tobe_removed.append(rel)
                elif rel.subj_ent in merged_cxrgraph_ents:
                    # subj need to be replaced
                    rel.obj_ent.chain_info[rel.label]["from"].remove(rel.subj_ent)
                    rel.obj_ent.chain_info[rel.label]["from"].append(new_ent)
                    rel.subj_ent = new_ent
                elif rel.obj_ent in merged_cxrgraph_ents:
                    rel.subj_ent.chain_info[rel.label]["to"].remove(rel.obj_ent)
                    rel.subj_ent.chain_info[rel.label]["to"].append(new_ent)
                    rel.obj_ent = new_ent

            for rel in rel_objs_tobe_removed:
                rel_list.remove(rel)

    assert len(rel_list) == len(set(rel_list)), f"{rel_list}"
    return ent_list, rel_list


def get_label_inheritance(cxrgraph_ents):
    candi_labels = [ent.label for ent in cxrgraph_ents]
    if "Observation-Absent" in candi_labels:
        return "Observation-Absent"
    elif "Observation-Uncertain" in candi_labels:
        return "Observation-Uncertain"
    elif "Observation-Present" in candi_labels:
        return "Observation-Present"
    elif "Anatomy" in candi_labels:
        return "Anatomy"
    else:
        return "Location-Attribute"


def get_attr_inheritance(cxrgraph_ents):
    candi_attr_normal = [ent.attr_normal for ent in cxrgraph_ents]
    candi_attr_action = [ent.attr_action for ent in cxrgraph_ents]
    candi_attr_change = [ent.attr_change for ent in cxrgraph_ents]
    assert all([i[0].istitle() for i in candi_attr_change]), f"{candi_attr_change} {candi_attr_normal} {candi_attr_action}"

    output_attr = {"normality": "NA", "action": "NA", "change": "NA"}
    if "Normal" in candi_attr_normal:
        output_attr["normality"] = "Normal"
    elif "Abnormal" in candi_attr_normal:
        output_attr["normality"] = "Abnormal"

    if "Essential" in candi_attr_action:
        output_attr["action"] = "Essential"
    elif "Removable" in candi_attr_action:
        output_attr["action"] = "Removable"

    if "Positive" in candi_attr_change:
        output_attr["change"] = "Positive"
    elif "Negative" in candi_attr_change:
        output_attr["change"] = "Negative"
    elif "Unchanged" in candi_attr_change:
        output_attr["change"] = "Unchanged"

    return output_attr


def check_span_relation(ent_a_indices, ent_b_indices):
    if ent_a_indices[1] <= ent_b_indices[0]:
        return "before"
    elif ent_a_indices[0] >= ent_b_indices[1]:
        return "after"
    elif ent_a_indices[0] == ent_b_indices[0] and ent_a_indices[1] == ent_b_indices[1]:
        return "equal"
    elif ent_a_indices[0] <= ent_b_indices[0] and ent_b_indices[1] <= ent_a_indices[1]:
        return "contain"
    elif ent_b_indices[0] <= ent_a_indices[0] and ent_a_indices[1] <= ent_b_indices[1]:
        return "inside"
    else:
        return "overlap"


if __name__ == "__main__":
    ds_final = load_from_disk("/home/yuxiang/liao/workspace/arrg_img2text/dataset_cache/clipbase_rbg224")
    sent_graph_repr = []
    for doc in tqdm(ds_final["test"]):
        for split_sent_idx, (cxrgraph_ent, cxrgraph_rel, cxrgraph_attr, radlex) in enumerate(zip(doc["cxrgraph_ent"], doc["cxrgraph_rel"], doc["cxrgraph_attr"], doc["radlex"])):

            sent_repr = SentenceRepresentation(doc_key=doc["doc_key"], sent_id=split_sent_idx, sent_text=doc["split_sents"])

            # resolve ent and rel from json
            ent_list, rel_list = resolve_ent_rel(split_sent_idx, cxrgraph_ent, cxrgraph_rel, cxrgraph_attr, radlex)

            linked_graphs = []
            visited_ents = set()
            for ent in ent_list:
                if ent not in visited_ents:
                    sent_ents = []
                    search_linked_ents(ent, visited_ents, sent_ents)
                    sent_graph = LinkedGraph(sent_ents)
                    sent_graph.get_involved_rels(rel_list)
                    linked_graphs.append(sent_graph)

            sent_repr.set_sent_repr(linked_graphs)
