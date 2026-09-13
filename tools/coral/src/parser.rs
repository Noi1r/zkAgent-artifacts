use pest::iterators::Pair;
use pest_meta::ast::Expr;
use pest_meta::ast::Expr::*;
use pest_meta::ast::RuleType;
use pest_meta::optimizer;
use pest_meta::parser::{self, Rule};
use pest_vm::Vm;
use petgraph::Direction;
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GrammarGraphNode {
    pub node_type: String,

    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct LcrsGraphNode {
    pub id: usize,
    pub rule_name: String,
    pub parent_id: Option<usize>,
    pub is_terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeType {
    Child,
    Sibling,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GrammarGraph {
    pub graph: DiGraph<GrammarGraphNode, ()>,
    pub lcrs_tree: Graph<LcrsGraphNode, EdgeType>,
    pub rules: HashMap<String, Vec<Vec<String>>>,
    pub np: HashMap<String, (Vec<String>, String)>,
    pub atom: Vec<String>,
    pub np_rule_names: HashSet<String>,
    pub max_rule_size: usize,
    pub rule_count: usize,
    pub rule_names: HashMap<String, Expr>,
    pub max_np_rule_size: usize,
}

impl Default for GrammarGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarGraph {
    pub fn new() -> Self {
        GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        }
    }

    #[allow(dead_code)]
    pub fn parse_text_and_build_graph(
        &mut self,
        grammar: &str,
        input_text: &str,
    ) -> Result<(), String> {
        let vm = self.compile_grammar(grammar)?;

        let pairs = vm.parse("root", input_text).map_err(|e| e.to_string())?;

        for pair in pairs {
            self.construct_parse_tree_node(pair, None);
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn compile_grammar(&mut self, grammar: &str) -> Result<Vm, String> {
        let pairs = parser::parse(Rule::grammar_rules, grammar).map_err(|e| e.to_string())?;

        let mut rules_map = parser::consume_rules(pairs).map_err(|errors| {
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })?;

        for rule in &rules_map {
            self.rule_names.insert(rule.name.clone(), rule.expr.clone());
        }

        self.transform_rules(&mut rules_map);

        self.create_table_vectors(&mut rules_map);

        let optimized = optimizer::optimize(rules_map);
        Ok(Vm::new(optimized))
    }

    #[allow(dead_code)]
    pub fn construct_parse_tree_node<R: pest::RuleType + ToString>(
        &mut self,
        pair: Pair<'_, R>,
        parent_index: Option<NodeIndex>,
    ) -> NodeIndex {
        let mut temp_child: Option<String> = None;
        let node_type = pair.as_rule().to_string();
        let value = if pair.clone().into_inner().peek().is_none() {
            if *pair.as_str() != node_type {
                temp_child = Some(pair.as_str().to_string());
                None
            } else {
                Some(pair.as_str().to_string())
            }
        } else {
            None
        };

        let node_index = self.graph.add_node(GrammarGraphNode {
            node_type: node_type.clone(),
            value: value.clone(),
        });

        if let Some(p_index) = parent_index {
            self.graph.add_edge(p_index, node_index, ());
        }

        if temp_child.is_some() {
            let temp_child_node_index = self.graph.add_node(GrammarGraphNode {
                node_type: temp_child.clone().unwrap(),
                value: temp_child,
            });
            self.graph.add_edge(node_index, temp_child_node_index, ());
        }

        if pair.clone().into_inner().peek().is_some() {
            for inner_pair in pair.into_inner() {
                self.construct_parse_tree_node(inner_pair, Some(node_index));
            }
        }


        node_index
    }

    fn transform_rules(&mut self, rules: &mut Vec<pest_meta::ast::Rule>) {
        let mut new_rules = HashMap::new();

        for rule in rules.iter_mut() {
            rule.expr = Self::transform_expr(&rule.expr, &mut new_rules);
        }

        for (name, expr) in new_rules {
            rules.push(pest_meta::ast::Rule {
                ty: pest_meta::ast::RuleType::Normal,
                name,
                expr,
            });
        }
    }

    fn create_table_vectors(&mut self, transformed_rules: &mut Vec<pest_meta::ast::Rule>) {
        self.rules.clear();
        let mut negpred_count = 0;
        let mut special_rules = Vec::new();
        let mut atomic_rules = Vec::new();

        for rule in transformed_rules.iter_mut() {
            let mut completed_rule = Vec::new();
            let mut rule_deques: Vec<VecDeque<String>> = Vec::new();

            if matches!(rule.ty, RuleType::Atomic) {
                rule.ty = RuleType::CompoundAtomic;
                self.atom.push(rule.name.clone());
            }

            self.process_expr(
                &rule.name,
                &mut rule.expr,
                VecDeque::new(),
                &mut rule_deques,
                &mut special_rules,
                &mut negpred_count,
            );

            if rule.name.starts_with("terminal_") {
                completed_rule.append(&mut Self::expand_terminals(&rule.name));

                for variant in completed_rule.clone() {
                    let entry = self.rules.entry(rule.name.clone()).or_default();

                    if !entry.contains(&variant) {
                        entry.push(variant);
                    }
                }
            } else {
                for deque in rule_deques {
                    let mut variant = deque.into_iter().collect::<Vec<String>>();
                    variant.push(rule.name.clone());
                    self.max_rule_size = max(self.max_rule_size, variant.len());

                    let entry = self.rules.entry(rule.name.clone()).or_default();

                    if !entry.contains(&variant) {
                        entry.push(variant.clone());
                    }
                }
            }
        }
        transformed_rules.append(&mut atomic_rules);
        transformed_rules.append(&mut special_rules);

        for rule_variants in self.rules.values_mut() {
            for variant in rule_variants.iter_mut() {
                while variant.len() < self.max_rule_size {
                    variant.push(String::new());
                }
                self.rule_count += 1;
            }
        }
    }

    fn modify_negpred_rule(
        &mut self,
        expr: &mut Expr,
        special_rules: &mut Vec<pest_meta::ast::Rule>,
        rule_deques: &mut Vec<VecDeque<String>>,
        negpred_count: &mut i32,
    ) -> String {
        let mut special_rule_name = String::new();

        if let Expr::Seq(a, _) = expr
            && let Expr::NegPred(_) = &**a
        {
            special_rule_name = format!("special{}", negpred_count);
            *negpred_count += 1;
            special_rules.push(pest_meta::ast::Rule {
                name: special_rule_name.clone(),
                ty: RuleType::Normal,
                expr: expr.clone(),
            });
            *expr = Expr::Ident(special_rule_name.clone());

            for rule in rule_deques.iter_mut() {
                while let Some(pos) = rule.iter().position(|r| r.starts_with("NegPred")) {
                    if pos > 0 {
                        rule.remove(pos - 1);
                        rule.remove(pos - 1);
                    }

                    rule.insert(pos - 1, special_rule_name.clone());
                }
            }
        }
        special_rule_name
    }

    fn expand_terminals(rule_name: &str) -> Vec<Vec<String>> {
        let mut expanded_rules = Vec::new();

        let ranges = match rule_name {
            "terminal_ASCII_DIGIT" => vec!['0'..='9'],
            "terminal_ASCII_NONZERO_DIGIT" => vec!['1'..='9'],
            "terminal_ASCII_BIN_DIGIT" => vec!['0'..='1'],
            "terminal_ASCII_OCT_DIGIT" => vec!['0'..='7'],
            "terminal_ASCII_HEX_DIGIT" => vec!['0'..='9', 'a'..='f', 'A'..='F'],
            "terminl_ASCII_ALPHA_LOWER" => vec!['a'..='z'],
            "terminal_ASCII_ALPHA_UPPER" => vec!['A'..='Z'],
            "terminal_ASCII_ALPHA" => vec!['a'..='z', 'A'..='Z'],
            "terminal_ASCII_ALPHANUMERIC" => vec!['a'..='z', 'A'..='Z', '0'..='9'],
            "terminal_ASCII" => vec!['\x00'..='\x7F'],

            _ => vec![],
        };

        if rule_name == "terminal_NEWLINE" {
            return vec![
                vec!["\n".to_string(), rule_name.to_string()],
                vec!["\r\n".to_string(), rule_name.to_string()],
                vec!["\r".to_string(), rule_name.to_string()],
            ];
        }

        for range in ranges {
            for ch in range {
                let mut new_rule = VecDeque::new();
                new_rule.push_back(ch.to_string());
                new_rule.push_back(rule_name.to_string());
                expanded_rules.push(new_rule.into_iter().collect::<Vec<String>>());
            }
        }
        expanded_rules
    }

    fn expand_terminal_negpred(&self, terminal_name: &str) -> Vec<String> {
        match terminal_name {
            "ASCII_DIGIT" => ('0'..='9').map(|c| c.to_string()).collect(),
            "ASCII_NONZERO_DIGIT" => ('1'..='9').map(|c| c.to_string()).collect(),
            "ASCII_BIN_DIGIT" => ('0'..='1').map(|c| c.to_string()).collect(),
            "ASCII_OCT_DIGIT" => ('0'..='7').map(|c| c.to_string()).collect(),
            "ASCII_HEX_DIGIT" => {
                let mut chars: Vec<String> = ('0'..='9').map(|c| c.to_string()).collect();
                chars.extend(('a'..='f').map(|c| c.to_string()));
                chars.extend(('A'..='F').map(|c| c.to_string()));
                chars
            }
            "ASCII_ALPHA_LOWER" => ('a'..='z').map(|c| c.to_string()).collect(),
            "ASCII_ALPHA_UPPER" => ('A'..='Z').map(|c| c.to_string()).collect(),
            "ASCII_ALPHA" => {
                let mut chars: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
                chars.extend(('A'..='Z').map(|c| c.to_string()));
                chars
            }
            "ASCII_ALPHANUMERIC" => {
                let mut chars: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
                chars.extend(('A'..='Z').map(|c| c.to_string()));
                chars.extend(('0'..='9').map(|c| c.to_string()));
                chars
            }
            "ASCII" => ('\x00'..='\x7F').map(|c| c.to_string()).collect(),
            "NEWLINE" => vec!["\n".to_string(), "\r\n".to_string(), "\r".to_string()],
            _ => vec![],
        }
    }

    fn collect_strings_from_negpred(
        &self,
        expr: &Expr,
        visited: &mut HashSet<String>,
    ) -> Vec<String> {
        match expr {
            Expr::Str(s) => vec![s.clone()],
            Expr::Seq(e1, e2) => {
                let left_strings = self.collect_strings_from_negpred(e1, visited);
                let right_strings = self.collect_strings_from_negpred(e2, visited);
                let mut combined = Vec::new();
                for ls in &left_strings {
                    for rs in &right_strings {
                        combined.push(format!("{}{}", ls, rs));
                    }
                }
                combined
            }
            Expr::Choice(e1, e2) => {
                let mut choices = self.collect_strings_from_negpred(e1, visited);
                choices.extend(self.collect_strings_from_negpred(e2, visited));
                choices
            }
            Expr::Ident(ident) => {
                if visited.contains(ident) {
                    return vec![];
                }
                visited.insert(ident.clone());

                if let Some(rule_expr) = self.rule_names.get(ident) {
                    let result = self.collect_strings_from_negpred(rule_expr, visited);
                    visited.remove(ident);
                    result
                } else {
                    let expanded = self.expand_terminal_negpred(ident);
                    visited.remove(ident);
                    expanded
                }
            }
            Expr::Insens(s) => {
                let mut variants = Vec::new();
                variants.push(s.clone());
                variants.push(s.to_uppercase());
                variants
            }
            _ => Vec::new(),
        }
    }

    fn process_expr(
        &mut self,
        rule_name: &str,
        expr: &mut Expr,
        current_path: VecDeque<String>,
        rule_deques: &mut Vec<VecDeque<String>>,
        special_rules: &mut Vec<pest_meta::ast::Rule>,
        negpred_count: &mut i32,
    ) {
        match expr {
            Expr::Seq(seq1, seq2) => {
                let mut first_part: Vec<VecDeque<String>> = vec![];
                self.process_expr(
                    rule_name,
                    seq1,
                    current_path.clone(),
                    &mut first_part,
                    special_rules,
                    negpred_count,
                );

                for item in first_part {
                    self.process_expr(
                        rule_name,
                        seq2,
                        item,
                        rule_deques,
                        special_rules,
                        negpred_count,
                    );
                }

                let mut exclude = Vec::new();

                let mut seq2_string = String::new();
                let mut has_negpred = false;

                {
                    if let Expr::NegPred(first) = &**seq1 {
                        has_negpred = true;

                        let mut visited = HashSet::new();
                        exclude = self.collect_strings_from_negpred(first.as_ref(), &mut visited);

                        seq2_string = seq2.to_string();
                    }
                }

                if has_negpred {
                    let key =
                        self.modify_negpred_rule(expr, special_rules, rule_deques, negpred_count);
                    self.np.insert(key, (exclude.clone(), seq2_string.clone()));

                    let exclude_size = exclude.len();
                    if exclude_size > self.max_np_rule_size {
                        self.max_np_rule_size = exclude_size;
                    }

                    for special_rule in special_rules.iter() {
                        if !self.np_rule_names.contains(&special_rule.name) {
                            let rule_vector = vec![seq2_string.clone(), special_rule.name.clone()];

                            let entry = self.rules.entry(special_rule.name.clone()).or_default();

                            if !entry.contains(&rule_vector) {
                                entry.push(rule_vector);
                            }

                            self.np_rule_names.insert(special_rule.name.clone());
                        }
                    }
                }
            }
            Expr::Choice(choice1, choice2) => {
                self.process_expr(
                    rule_name,
                    choice1,
                    current_path.clone(),
                    rule_deques,
                    special_rules,
                    negpred_count,
                );
                self.process_expr(
                    rule_name,
                    choice2,
                    current_path.clone(),
                    rule_deques,
                    special_rules,
                    negpred_count,
                );
            }
            Expr::Ident(ident) => {
                let ident_str = ident.clone();

                if ident_str == "SOI" {
                    let rule_name = format!("terminal_{}", ident_str);
                    let rule_vector = vec!["".to_string(), rule_name.clone()];

                    let entry = self.rules.entry(rule_name.clone()).or_default();

                    if !entry.contains(&rule_vector) {
                        entry.push(rule_vector);
                    }
                } else if ident_str == "EOI" {
                    let rule_vector = vec!["".to_string(), ident_str.clone()];

                    let entry = self.rules.entry(ident_str.clone()).or_default();

                    if !entry.contains(&rule_vector) {
                        entry.push(rule_vector);
                    }
                }

                let mut new_current_path = current_path.clone();
                new_current_path.push_front(ident_str.clone());
                rule_deques.push(new_current_path);
            }
            Expr::Str(s) => {
                let mut new_current_path = current_path.clone();
                new_current_path.push_front(s.clone());
                rule_deques.push(new_current_path);
            }
            Expr::Insens(s) => {
                let new_current_path = current_path.clone();
                let uppercase_version = s.to_uppercase();

                let mut path_with_original = new_current_path.clone();
                path_with_original.push_front(s.clone());
                rule_deques.push(path_with_original);

                let mut path_with_uppercase = new_current_path;
                path_with_uppercase.push_front(uppercase_version);
                rule_deques.push(path_with_uppercase);
            }
            Expr::RepExact(expr, count) => {
                let mut new_current_path = current_path.clone();
                for _ in 0..*count {
                    if let Expr::Ident(ident) = &**expr {
                        new_current_path.push_front(ident.clone());
                    } else {
                        let mut temp_table: Vec<VecDeque<String>> = vec![];
                        self.process_expr(
                            rule_name,
                            expr,
                            VecDeque::new(),
                            &mut temp_table,
                            special_rules,
                            negpred_count,
                        );
                        for item in temp_table {
                            let mut extended_path = new_current_path.clone();
                            extended_path.append(&mut item.clone());
                            rule_deques.push(extended_path);
                        }
                    }
                }
            }
            _ => {
                let mut new_current_path = current_path.clone();
                new_current_path.push_front(format!("{:?}", expr));
                rule_deques.push(new_current_path);
            }
        }
    }

    fn transform_expr(
        expr: &pest_meta::ast::Expr,
        new_rules: &mut HashMap<String, pest_meta::ast::Expr>,
    ) -> pest_meta::ast::Expr {
        match expr {
            Str(terminal) => {
                if terminal.len() > 1 {
                    let lhs = terminal[0..1].to_string();
                    let rhs = terminal[1..].to_string();
                    Seq(
                        Box::new(Self::transform_expr(&Str(lhs), new_rules)),
                        Box::new(Self::transform_expr(&Str(rhs), new_rules)),
                    )
                } else if terminal != " "
                    && terminal != "\t"
                    && terminal != "\n"
                    && terminal != "\r\n"
                {
                    let rule_name = terminal.clone();
                    new_rules
                        .entry(rule_name.clone())
                        .or_insert_with(|| Str(terminal.clone()));
                    Ident(rule_name)
                } else {
                    Str(terminal.clone())
                }
            }
            Insens(terminal) => {
                let rule_name = format!("Insens_{}", terminal);
                new_rules
                    .entry(rule_name.clone())
                    .or_insert_with(|| Insens(terminal.clone()));
                Ident(rule_name)
            }
            Ident(terminal) if Self::is_terminal(terminal) => {
                let rule_name = format!("terminal_{}", terminal);
                new_rules
                    .entry(rule_name.clone())
                    .or_insert_with(|| Ident(terminal.clone()));
                Ident(rule_name)
            }
            Seq(lhs, rhs) => Seq(
                Box::new(Self::transform_expr(lhs, new_rules)),
                Box::new(Self::transform_expr(rhs, new_rules)),
            ),
            Choice(lhs, rhs) => Choice(
                Box::new(Self::transform_expr(lhs, new_rules)),
                Box::new(Self::transform_expr(rhs, new_rules)),
            ),
            Range(start, end) => {
                let range_rule_name = format!("range_{}_{}", start, end);
                if !new_rules.contains_key(&range_rule_name) {
                    let range_characters = (start.chars().next().unwrap()
                        ..=end.chars().next().unwrap())
                        .map(|c| Str(c.to_string()))
                        .collect::<Vec<_>>();
                    let combined_expr = range_characters
                        .into_iter()
                        .reduce(|a, b| Choice(Box::new(a), Box::new(b)))
                        .unwrap();
                    new_rules.insert(range_rule_name.clone(), combined_expr);
                }
                Ident(range_rule_name)
            }
            Opt(inner_expr) => {
                let transformed_inner = Self::transform_expr(inner_expr, new_rules);
                let opt_rule_name = Self::generate_unique_id(&transformed_inner);
                if !new_rules.contains_key(&opt_rule_name) {
                    let optional_expr = Choice(
                        Box::new(transformed_inner.clone()),
                        Box::new(Str("".to_string())),
                    );
                    new_rules.insert(opt_rule_name.clone(), optional_expr);
                }
                Ident(opt_rule_name)
            }
            Rep(inner_expr) => {
                let rep_rule_name = format!("Rep_{}", Self::generate_unique_id(inner_expr));

                if !new_rules.contains_key(&rep_rule_name) {
                    let transformed_inner = Self::transform_expr(inner_expr, new_rules);
                    let repeated_expr = Seq(
                        Box::new(transformed_inner.clone()),
                        Box::new(Ident(rep_rule_name.clone())),
                    );
                    let rep_expr = Choice(Box::new(repeated_expr), Box::new(Str("".to_string())));

                    new_rules.insert(rep_rule_name.clone(), rep_expr);
                }
                Ident(rep_rule_name)
            }
            RepOnce(inner_expr) => {
                let rep_once_rule_name =
                    format!("RepOnce_{}", Self::generate_unique_id(inner_expr));

                if !new_rules.contains_key(&rep_once_rule_name) {
                    let transformed_inner = Self::transform_expr(inner_expr, new_rules);
                    let repeated_expr = Seq(
                        Box::new(Self::transform_expr(&transformed_inner, new_rules)),
                        Box::new(Self::transform_expr(
                            &Rep(Box::new(transformed_inner.clone())),
                            new_rules,
                        )),
                    );

                    new_rules.insert(rep_once_rule_name.clone(), repeated_expr);
                }

                Ident(rep_once_rule_name)
            }
            _ => expr.clone(),
        }
    }

    pub fn is_terminal(name: &str) -> bool {
        matches!(
            name,
            "ANY"
            | "SOI"
            | "PEEK"
            | "PEEK_ALL"
            | "POP"
            | "POP_ALL"
            | "DROP"
            | "ASCII_DIGIT"
            | "ASCII_NONZERO_DIGIT"
            | "ASCII_BIN_DIGIT"
            | "ASCII_OCT_DIGIT"
            | "ASCII_HEX_DIGIT"
            | "ASCII_ALPHA_LOWER"
            | "ASCII_ALPHA_UPPER"
            | "ASCII_ALPHA"
            | "ASCII_ALPHANUMERIC"
            | "ASCII"
            | "NEWLINE"
        )
    }

    pub fn generate_unique_id(expr: &pest_meta::ast::Expr) -> String {
        use pest_meta::ast::Expr::*;
        match expr {
            Str(s) => s.to_string(),
            NegPred(s) => format!("NegPred_{}", Self::generate_unique_id(s)),
            Ident(s) => format!("Ident_{}", s),
            Seq(left, right) => format!(
                "Seq_{}_{}",
                Self::generate_unique_id(left),
                Self::generate_unique_id(right)
            ),
            Choice(left, right) => {
                format!(
                    "Choice_{}_{}",
                    Self::generate_unique_id(left),
                    Self::generate_unique_id(right)
                )
            }
            Opt(inner) => format!("Opt_{}", Self::generate_unique_id(inner)),
            Rep(inner) => format!("Rep_{}", Self::generate_unique_id(inner)),
            RepOnce(inner) => format!("RepOnce_{}", Self::generate_unique_id(inner)),
            Range(start, end) => format!("Range_{}_{}", start, end),
            _ => "Unsupported_Expr_Type".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn parse_and_convert_lcrs(&mut self) {
        let mut node_map: HashMap<usize, NodeIndex> = HashMap::new();
        for node_index in self.graph.node_indices() {
            let original_node = &self.graph[node_index];
            let rule_name = if original_node.value.is_some() {
                original_node
                    .value
                    .clone()
                    .unwrap_or_else(|| "Default value".to_string())
            } else {
                original_node.node_type.clone()
            };
            let is_terminal = original_node.value.is_some();
            let node_data = LcrsGraphNode {
                id: node_index.index(),
                rule_name,
                parent_id: None,
                is_terminal,
            };
            let lcrs_node = self.lcrs_tree.add_node(node_data.clone());
            node_map.insert(node_data.id, lcrs_node);
        }

        for node_index in self.graph.node_indices() {
            let node_data = node_index.index();
            if let Some(&parent_lcrs_node) = node_map.get(&node_data) {
                let mut prev_sibling: Option<NodeIndex> = None;
                let neighbors: Vec<NodeIndex> = self
                    .graph
                    .neighbors_directed(node_index, Direction::Outgoing)
                    .collect();

                for &neighbor in neighbors.iter().rev() {
                    let neighbor_data = neighbor.index();
                    if let Some(&child_lcrs_node) = node_map.get(&neighbor_data) {
                        if prev_sibling.is_none() {
                            self.lcrs_tree.add_edge(
                                parent_lcrs_node,
                                child_lcrs_node,
                                EdgeType::Child,
                            );
                        } else {
                            self.lcrs_tree.add_edge(
                                prev_sibling.unwrap(),
                                child_lcrs_node,
                                EdgeType::Sibling,
                            );
                        }
                        prev_sibling = Some(child_lcrs_node);
                        self.lcrs_tree[child_lcrs_node].parent_id = Some(node_data);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_node(&self, id: usize) -> Option<&LcrsGraphNode> {
        self.lcrs_tree.node_indices().find_map(|i| {
            if self.lcrs_tree[i].id == id {
                Some(&self.lcrs_tree[i])
            } else {
                None
            }
        })
    }

    #[allow(dead_code)]
    pub fn get_all_siblings(&self, id: usize) -> Vec<LcrsGraphNode> {
        let mut siblings = Vec::new();

        if let Some(node_index) = self
            .lcrs_tree
            .node_indices()
            .find(|&i| self.lcrs_tree[i].id == id)
        {
            let parent_id = self.lcrs_tree[node_index].parent_id;
            if parent_id != Some(usize::MAX) {
                if let Some(parent_id) = parent_id
                    && let Some(parent_index) = self
                        .lcrs_tree
                        .node_indices()
                        .find(|&i| self.lcrs_tree[i].id == parent_id)
                {
                    for edge in self
                        .lcrs_tree
                        .edges_directed(parent_index, Direction::Outgoing)
                    {
                        if edge.weight() == &EdgeType::Child {
                            let child_index = edge.target();
                            if self.lcrs_tree[child_index].id != id {
                                siblings.push(self.lcrs_tree[child_index].clone());
                            }
                            let mut sibling_index = Some(child_index);

                            while let Some(current_index) = sibling_index {
                                sibling_index = self
                                    .lcrs_tree
                                    .edges_directed(current_index, Direction::Outgoing)
                                    .filter_map(|next_edge| {
                                        if next_edge.weight() == &EdgeType::Sibling {
                                            Some(next_edge.target())
                                        } else {
                                            None
                                        }
                                    })
                                    .next();
                                if let Some(next_index) = sibling_index
                                    && self.lcrs_tree[next_index].id != id
                                {
                                    siblings.push(self.lcrs_tree[next_index].clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        siblings
    }

    #[allow(dead_code)]
    pub fn get_all_children(&self, id: usize) -> Vec<LcrsGraphNode> {
        let mut children = Vec::new();

        if let Some(node_index) = self
            .lcrs_tree
            .node_indices()
            .find(|&i| self.lcrs_tree[i].id == id)
        {
            let mut child_index = self
                .lcrs_tree
                .edges_directed(node_index, Direction::Outgoing)
                .find(|edge| edge.weight() == &EdgeType::Child)
                .map(|edge| edge.target());

            while let Some(current_index) = child_index {
                children.push(self.lcrs_tree[current_index].clone());
                child_index = self
                    .lcrs_tree
                    .edges_directed(current_index, Direction::Outgoing)
                    .filter_map(|edge| {
                        if edge.weight() == &EdgeType::Sibling {
                            Some(edge.target())
                        } else {
                            None
                        }
                    })
                    .next();
            }
        }
        children
    }

    #[allow(dead_code)]
    pub fn get_specific_nodes(&self) {
        let node_id = 0;

        self.get_node(node_id);

        self.get_all_siblings(node_id);

        self.get_all_children(node_id);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pest_meta::ast::Expr;
    use std::collections::HashMap;

    #[test]
    fn test_get_node() {
        let mut graph = GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        };

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 1,
            rule_name: "Root".to_string(),
            parent_id: None,
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 2,
            rule_name: "Child1".to_string(),
            parent_id: Some(1),
            is_terminal: true,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 3,
            rule_name: "Child2".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 4,
            rule_name: "Child3".to_string(),
            parent_id: Some(2),
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 5,
            rule_name: "Child4".to_string(),
            parent_id: Some(2),
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 6,
            rule_name: "Child5".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 7,
            rule_name: "Child6".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });

        graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 8,
            rule_name: "Child7".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });

        assert_eq!(graph.get_node(1).unwrap().id, 1);
        assert_eq!(graph.get_node(2).unwrap().id, 2);
        assert_eq!(graph.get_node(3).unwrap().id, 3);
        assert_eq!(graph.get_node(4).unwrap().id, 4);
        assert_eq!(graph.get_node(5).unwrap().id, 5);
        assert_eq!(graph.get_node(6).unwrap().id, 6);
        assert_eq!(graph.get_node(7).unwrap().id, 7);
        assert_eq!(graph.get_node(8).unwrap().id, 8);

        assert!(graph.get_node(9).is_none());
    }

    #[test]
    fn test_get_all_siblings() {
        let mut graph = GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        };

        let root_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 1,
            rule_name: "Root".to_string(),
            parent_id: None,
            is_terminal: false,
        });

        let child1_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 2,
            rule_name: "Child1".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });
        let child2_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 3,
            rule_name: "Child2".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });

        graph
            .lcrs_tree
            .add_edge(root_index, child1_index, EdgeType::Child);
        graph
            .lcrs_tree
            .add_edge(child1_index, child2_index, EdgeType::Sibling);

        let child3_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 4,
            rule_name: "Child3".to_string(),
            parent_id: Some(2),
            is_terminal: false,
        });
        let child4_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 5,
            rule_name: "Child4".to_string(),
            parent_id: Some(2),
            is_terminal: false,
        });

        graph
            .lcrs_tree
            .add_edge(child1_index, child3_index, EdgeType::Child);
        graph
            .lcrs_tree
            .add_edge(child3_index, child4_index, EdgeType::Sibling);

        let child5_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 6,
            rule_name: "Child5".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });
        let child6_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 7,
            rule_name: "Child6".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });
        let child7_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 8,
            rule_name: "Child7".to_string(),
            parent_id: Some(3),
            is_terminal: false,
        });

        graph
            .lcrs_tree
            .add_edge(child2_index, child5_index, EdgeType::Child);
        graph
            .lcrs_tree
            .add_edge(child5_index, child6_index, EdgeType::Sibling);
        graph
            .lcrs_tree
            .add_edge(child6_index, child7_index, EdgeType::Sibling);

        let siblings = graph.get_all_siblings(6);

        assert_eq!(siblings.len(), 2);
        assert!(siblings.iter().any(|n| n.id == 7));
        assert!(siblings.iter().any(|n| n.id == 8));
    }

    #[test]
    fn test_get_all_children() {
        let mut graph = GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        };

        let parent_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 1,
            rule_name: "Parent".to_string(),
            parent_id: None,
            is_terminal: false,
        });

        let child1_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 2,
            rule_name: "Child1".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });
        let child2_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 3,
            rule_name: "Child2".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });
        let child3_index = graph.lcrs_tree.add_node(LcrsGraphNode {
            id: 4,
            rule_name: "Child3".to_string(),
            parent_id: Some(1),
            is_terminal: false,
        });

        graph
            .lcrs_tree
            .add_edge(parent_index, child1_index, EdgeType::Child);
        graph
            .lcrs_tree
            .add_edge(child1_index, child2_index, EdgeType::Sibling);
        graph
            .lcrs_tree
            .add_edge(child2_index, child3_index, EdgeType::Sibling);

        let children = graph.get_all_children(1);

        assert_eq!(children.len(), 3);
        assert!(children.iter().any(|n| n.id == 2));
        assert!(children.iter().any(|n| n.id == 3));
        assert!(children.iter().any(|n| n.id == 4));
    }

    #[test]
    fn test_parse_and_convert_lcrs() {
        let mut graph = GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        };

        let root = graph.graph.add_node(GrammarGraphNode {
            value: None,
            node_type: "Root".to_string(),
        });
        let child1 = graph.graph.add_node(GrammarGraphNode {
            value: None,
            node_type: "Child1".to_string(),
        });
        let child2 = graph.graph.add_node(GrammarGraphNode {
            value: None,
            node_type: "Child2".to_string(),
        });

        graph.graph.add_edge(root, child1, ());
        graph.graph.add_edge(root, child2, ());

        let child3 = graph.graph.add_node(GrammarGraphNode {
            value: None,
            node_type: "Child3".to_string(),
        });
        graph.graph.add_edge(child1, child3, ());

        graph.parse_and_convert_lcrs();

        assert_eq!(
            graph.lcrs_tree.node_count(),
            4,
            "There should be four nodes in the LCRS tree."
        );

        let nodes: Vec<_> = graph
            .lcrs_tree
            .raw_nodes()
            .iter()
            .map(|n| n.weight.rule_name.clone())
            .collect();
        assert!(nodes.contains(&"Root".to_string()));
        assert!(nodes.contains(&"Child1".to_string()));
        assert!(nodes.contains(&"Child2".to_string()));
        assert!(nodes.contains(&"Child3".to_string()));

        let root_node_index = nodes.iter().position(|n| n == "Root").unwrap();
        let child1_node_index = nodes.iter().position(|n| n == "Child1").unwrap();
        let child2_node_index = nodes.iter().position(|n| n == "Child2").unwrap();
        let child3_node_index = nodes.iter().position(|n| n == "Child3").unwrap();

        assert!(
            graph
                .lcrs_tree
                .find_edge(
                    NodeIndex::new(root_node_index),
                    NodeIndex::new(child1_node_index)
                )
                .is_some(),
            "Root should be connected to Child1 as a child."
        );
        assert!(
            graph
                .lcrs_tree
                .find_edge(
                    NodeIndex::new(child1_node_index),
                    NodeIndex::new(child2_node_index)
                )
                .is_some(),
            "Child1 should be connected to Child2 as a sibling."
        );
        assert!(
            graph
                .lcrs_tree
                .find_edge(
                    NodeIndex::new(child1_node_index),
                    NodeIndex::new(child3_node_index)
                )
                .is_some(),
            "Child1 should be connected to Child3 as a child."
        );
    }

    #[test]
    fn test_generate_unique_ids() {
        let cases = vec![
            (Ident("variable".to_string()), "Ident_variable"),
            (
                Seq(
                    Box::new(Str("first".to_string())),
                    Box::new(Ident("second".to_string())),
                ),
                "Seq_first_Ident_second",
            ),
            (
                Choice(
                    Box::new(Ident("left".to_string())),
                    Box::new(Ident("right".to_string())),
                ),
                "Choice_Ident_left_Ident_right",
            ),
            (Opt(Box::new(Str("optional".to_string()))), "Opt_optional"),
            (
                Rep(Box::new(Ident("repeat".to_string()))),
                "Rep_Ident_repeat",
            ),
            (
                RepOnce(Box::new(Ident("repeat_once".to_string()))),
                "RepOnce_Ident_repeat_once",
            ),
            (Range("a".to_string(), "z".to_string()), "Range_a_z"),
        ];

        for (expr, expected_id) in cases {
            assert_eq!(
                GrammarGraph::generate_unique_id(&expr),
                expected_id,
                "Failed on {:?}",
                expr
            );
        }
    }

    #[test]
    fn test_is_terminal() {
        let terminals = [
            "ANY",
            "SOI",
            "PEEK",
            "PEEK_ALL",
            "POP",
            "POP_ALL",
            "DROP",
            "ASCII_DIGIT",
            "ASCII_NONZERO_DIGIT",
            "ASCII_BIN_DIGIT",
            "ASCII_OCT_DIGIT",
            "ASCII_HEX_DIGIT",
            "ASCII_ALPHA_LOWER",
            "ASCII_ALPHA_UPPER",
            "ASCII_ALPHA",
            "ASCII_ALPHANUMERIC",
            "ASCII",
            "NEWLINE",
        ];

        for term in terminals.iter() {
            assert!(
                GrammarGraph::is_terminal(term),
                "Expected {} to be a terminal, but it was not recognized as one.",
                term
            );
        }

        let non_terminals = ["Example", "Test", "123"];

        for non_term in non_terminals.iter() {
            assert!(
                !GrammarGraph::is_terminal(non_term),
                "Expected {} not to be a terminal, but it was incorrectly recognized as one.",
                non_term
            );
        }
    }

    #[test]
    fn test_transform_ident() {
        let mut new_rules = HashMap::new();
        let expr = Ident("ANY".to_string());

        let result = GrammarGraph::transform_expr(&expr, &mut new_rules);

        assert_eq!(result, Ident("terminal_ANY".to_string()));
        assert!(
            new_rules.contains_key("terminal_ANY"),
            "New rule for 'terminal_ANY' should be added."
        );
    }

    #[test]
    fn test_transform_seq() {
        let mut new_rules = HashMap::new();
        let expr = Seq(
            Box::new(Ident("ANY".to_string())),
            Box::new(Ident("EOI".to_string())),
        );

        let expected = Seq(
            Box::new(Ident("terminal_ANY".to_string())),
            Box::new(Ident("EOI".to_string())),
        );

        let result = GrammarGraph::transform_expr(&expr, &mut new_rules);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_transform_choice() {
        let mut new_rules = HashMap::new();
        let expr = Choice(
            Box::new(Ident("ANY".to_string())),
            Box::new(Ident("EOI".to_string())),
        );

        let expected = Choice(
            Box::new(Ident("terminal_ANY".to_string())),
            Box::new(Ident("EOI".to_string())),
        );

        let result = GrammarGraph::transform_expr(&expr, &mut new_rules);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_transform_range() {
        let mut new_rules = HashMap::new();
        let expr = Range("a".to_string(), "z".to_string());

        GrammarGraph::transform_expr(&expr, &mut new_rules);
        let range_rule_name = "range_a_z";

        assert!(
            new_rules.contains_key(range_rule_name),
            "Range should be added to new_rules"
        );
        if let Some(Expr::Choice(..)) = new_rules.get(range_rule_name) {
            println!("Range expression correctly transformed into Choice structure.");
        } else {
            panic!(
                "Expected a Choice structure for the range, but found {:?}",
                new_rules.get(range_rule_name)
            );
        }
    }

    #[test]
    fn test_transform_opt() {
        let mut new_rules = HashMap::new();
        let expr = Opt(Box::new(Ident("ANY".to_string())));
        GrammarGraph::transform_expr(&expr, &mut new_rules);

        let transformed_inner = Ident("terminal_ANY".to_string());
        let opt_rule_name = GrammarGraph::generate_unique_id(&transformed_inner);

        assert!(
            new_rules.contains_key(&opt_rule_name),
            "Optional expression should be added to new_rules under: {}",
            opt_rule_name
        );
        assert!(
            matches!(new_rules[&opt_rule_name], Choice(_, _)),
            "Optional expression should be a Choice type"
        );
    }

    #[test]
    fn test_transform_rep() {
        let mut new_rules = HashMap::new();
        let expr = Rep(Box::new(Ident("ANY".to_string())));
        let rep_rule_name = "Rep_Ident_ANY";
        GrammarGraph::transform_expr(&expr, &mut new_rules);
        println!("Generated rep_rule_name: {}", rep_rule_name);

        assert!(
            new_rules.contains_key(rep_rule_name),
            "Rep expression should be added to new_rules"
        );
        assert!(
            matches!(new_rules[rep_rule_name], Choice(_, _)),
            "Rep expression should be a Choice type"
        );
    }

    #[test]
    fn test_transform_rep_once() {
        let mut new_rules = HashMap::new();
        let expr = RepOnce(Box::new(Ident("ANY".to_string())));
        let rep_once_rule_name = "RepOnce_Ident_ANY";
        GrammarGraph::transform_expr(&expr, &mut new_rules);

        assert!(
            new_rules.contains_key(rep_once_rule_name),
            "RepOnce expression should be added to new_rules"
        );
        assert!(
            matches!(new_rules[rep_once_rule_name], Seq(_, _)),
            "RepOnce expression should be a Seq type"
        );
    }

    #[test]
    fn test_transform_seq_break() {
        let grammar = "root = {SOI ~ ANY ~ EOI}";
        let input_text = "a";
        let mut grammar_graph = GrammarGraph::new();
        grammar_graph
            .parse_text_and_build_graph(&grammar, &input_text)
            .expect("Failed to parse input");
    }

    #[test]
    fn test_process_expr() {
        let mut graph = GrammarGraph {
            graph: DiGraph::new(),
            lcrs_tree: Graph::new(),
            rules: HashMap::new(),
            np: HashMap::new(),
            atom: Vec::new(),
            np_rule_names: HashSet::new(),
            max_rule_size: 0,
            rule_count: 0,
            rule_names: HashMap::new(),
            max_np_rule_size: 0,
        };
        let mut expr = Expr::Seq(
            Box::new(Expr::Seq(
                Box::new(Expr::Ident("terminal_SOI".to_string())),
                Box::new(Expr::Choice(
                    Box::new(Expr::Ident("object".to_string())),
                    Box::new(Expr::Ident("array".to_string())),
                )),
            )),
            Box::new(Expr::Ident("EOI".to_string())),
        );

        let mut rule_deques: Vec<VecDeque<String>> = Vec::new();
        let mut special_rules: Vec<pest_meta::ast::Rule> = Vec::new();
        let mut negpred_count = 0;
        let rule_name = "root";

        graph.process_expr(
            rule_name,
            &mut expr,
            VecDeque::new(),
            &mut rule_deques,
            &mut special_rules,
            &mut negpred_count,
        );

        let expected_results = vec![
            vec!["EOI", "object", "terminal_SOI"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>(),
            vec!["EOI", "array", "terminal_SOI"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>(),
        ];

        assert_eq!(rule_deques, expected_results);
    }

    #[test]
    fn test_expand_terminals() {
        let test_cases = vec![
            ("terminal_ASCII_DIGIT", vec!['0'..='9']),
            ("terminal_ASCII_NONZERO_DIGIT", vec!['1'..='9']),
            ("terminal_ASCII_BIN_DIGIT", vec!['0'..='1']),
            ("terminal_ASCII_OCT_DIGIT", vec!['0'..='7']),
            (
                "terminal_ASCII_HEX_DIGIT",
                vec!['0'..='9', 'a'..='f', 'A'..='F'],
            ),
            ("terminl_ASCII_ALPHA_LOWER", vec!['a'..='z']),
            ("terminal_ASCII_ALPHA_UPPER", vec!['A'..='Z']),
            ("terminal_ASCII_ALPHA", vec!['a'..='z', 'A'..='Z']),
            (
                "terminal_ASCII_ALPHANUMERIC",
                vec!['a'..='z', 'A'..='Z', '0'..='9'],
            ),
            ("terminal_ASCII", vec!['\x00'..='\x7F']),
        ];

        for (rule_name, expected_ranges) in test_cases {
            let expanded = GrammarGraph::expand_terminals(rule_name);
            let expected_count: usize = expected_ranges
                .iter()
                .map(|r| (*r.end() as usize) - (*r.start() as usize) + 1)
                .sum();

            assert_eq!(
                expanded.len(),
                expected_count,
                "Incorrect expansion count for {}",
                rule_name
            );
            for (vec, ch) in expanded
                .iter()
                .zip(expected_ranges.iter().flat_map(|r| r.clone()))
            {
                assert_eq!(
                    vec[0],
                    ch.to_string(),
                    "Character mismatch in expansion for {}",
                    rule_name
                );
            }
        }

        let newline_expanded = GrammarGraph::expand_terminals("terminal_NEWLINE");
        assert_eq!(newline_expanded.len(), 3, "Incorrect expansion for NEWLINE");
        assert!(newline_expanded.contains(&vec!["\n".to_string(), "terminal_NEWLINE".to_string()]));
        assert!(
            newline_expanded.contains(&vec!["\r\n".to_string(), "terminal_NEWLINE".to_string()])
        );
        assert!(newline_expanded.contains(&vec!["\r".to_string(), "terminal_NEWLINE".to_string()]));
    }
}
