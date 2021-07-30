

const InputField = (props) => {

    return (
        <>
            <label htmlFor="id_sample_size">No. of Words:</label>
            <input 
                id="id_sample_size"
                name="sample_size"
                type="text"
                placeholder="..."
                onFocus={(e) => e.target.placeholder = ""} 
                onBlur={(e) => e.target.placeholder = "..."}                        
            />
        </>
    )
}

export default InputField
