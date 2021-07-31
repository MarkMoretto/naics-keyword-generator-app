

const DataList = props => {
    const itemArray = props.items
    
    const listOptions = itemArray.map((i) => 
        <option value={i.value} />
    )

    return (
        <div>
            <datalist>
                {listOptions}
            </datalist>
        </div>
    )
}


export default DataList
