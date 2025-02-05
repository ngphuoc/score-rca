using Flux, JSON, Functors

struct OrderedShapley
    P
end
@functor OrderedShapley

function OrderedShapley()
    P = Set{Any}()
    OrderedShapley(P)
end

function powerset(x)
    s = collect(x)
    return chain.from_iterable(combinations(s, r) for r in 1:length(s + 1))
end

function _r(S, channel_index, touchpoint_index)
    a = []
    for journey, journey_set in indexed_journeys[length(S)]
        if touchpoint_index <= length(journey)
            v = if S == journey_set && journey[touchpoint_index - 1] == channel_index
                1 / journey.count(channel_index)
            else
                0
            end
            push!(a, v)
        end
    end

    return sum(a)
end

function _phi(channel_index, touchpoint_index)
    S_all = [set(S) for S in P_power if channel_index in S]
    score = 0
    print(
        f"Computing phi for channel {channel_index}, touchpoint {touchpoint_index}..."
    )
    for S in tqdm(S_all)
        score += _r(S, channel_index, touchpoint_index) / length(S)
    print(
        f"Attribution score for channel {channel_index}, touchpoint {touchpoint_index}: {score:.2f}"
    )
    print()
    return score
end

function attribute(journeys)
    P = set(chain(*journeys))
    print("Running Ordered Shapley Attribution Model...")
    print(f"Found {length(P)} unique channels!")
    P_power = collect(powerset(P))
    N = max([length(journey) for journey in journeys])
    print(f"Found {N} maximum touchpoints!")
    journeys = journeys
    indexed_journeys = {
        i: [(S, set(S)) for S in journeys if length(set(S)) == i]
        for i in 1:1, length(P + 1)
    }
    print(f"Proceeding to attribution computation...")
    print()
    return {j: [_phi(j, i) for i in 1:1, N + 1] for j in P}
end

struct SimplifiedShapley
end
@functor SimplifiedShapley

function powerset(x)
    s = collect(x)
    return chain.from_iterable(combinations(s, r) for r in 1:length(s + 1))
end

function _phi(channel_index)
    S_channel = [k for k in journeys.keys() if channel_index in k]
    score = 0
    print(f"Computing phi for channel {channel_index}...")
    for S in tqdm(S_channel)
        score += journeys[S] / length(S)
    print(f"Attribution score for channel {channel_index}: {score:.2f}")
    print()
    return score
end

function attribute(journeys)
    P = set(chain(*journeys))
    print("Running Simplified Shapley Attribution Model...")
    print(f"Found {length(P)} unique channels!")

    print("Computing journey statistics...")
    journeys = Counter([frozenset(journey) for journey in journeys])

    print(f"Computing attributions...")
    print()
    return {j: _phi(j) for j in P}
end

function test_shapley()
    journeys = JSON.parsefile("data/sample.json")
    o = OrderedShapley()
    result = o.attribute(journeys)

    print(f"Total value: {len(journeys)}")
    total = 0
    for k, v in result.items()
        vsum = sum(v)
        print(f"Channel {k}: {vsum}")
        total += vsum
        print(f"Total of attributed values: {total:.2f}")
    end
end

