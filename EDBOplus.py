import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    #import json
    import toml
    import glob
    import os, sys, shutil
    #import edboplus.edbo
    from edboplus.edbo.plus.optimizer_botorch import EDBOplus
    return EDBOplus, glob, mo, np, os, pd, shutil, toml


@app.cell
def _():
    button_dict = {}
    button_dict["rn_btn0"] = 0
    button_dict["rn_btn1"] = 0
    return


@app.cell
def _(glob, os):
    # search projects
    database_folder_name = "database"
    database_folder_path = database_folder_name+os.path.sep

    def get_projects(database_folder_name, database_folder_path):
        projects = glob.glob( os.path.join(database_folder_name,"*"))
        projects = [p.replace(database_folder_path,"") for p in projects]
        projects = ["new project"] + projects    
        return projects

    projects= get_projects(database_folder_name,database_folder_path)
    return database_folder_path, projects


@app.cell
def _(mo, projects):
    new_project_string = "new project"
    project_dropdown = mo.ui.dropdown(
        options=projects,
        value=new_project_string,
        #label="choose your project or start a new one:",
        #searchable=True,
    )
    project_name_dropdown = mo.md(
    """
    # Select Project

    ### Choose a project or start a new one: {project_dropdown}

    """   
    ).batch(project_dropdown=project_dropdown)
    project_name_dropdown
    return new_project_string, project_name_dropdown


@app.cell
def _(os, pd, toml):
    """
    - not sure if two classes are necessary.
    - pathing might be replacable with a decorator class
    - ...details not importatnt. Its a test. It needs to work.

    """


    class project_pathing():

        def __init__(self,project_name, database_folder_path,update_key="yield"):
            self.project_name = project_name
            self.database_folder_path = database_folder_path
            self.project_path = os.path.join(database_folder_path,project_name)
            self.edbo_filename = 'my_optimization.csv'
            self.edbo_filename2 = 'my_optimization_temp.csv'
            self.edbo_filepath = os.path.join(self.project_path,self.edbo_filename)
            self.update_key=update_key

        def save_toml(self,filename, toml_dict):
            filepath = os.path.join(self.project_path,filename)
            with open(filepath, "w") as toml_file:
                toml.dump(toml_dict, toml_file)        

        def load_toml(self,filename):
            filepath = os.path.join(self.project_path,filename)
            with open(filepath) as toml_file:
                toml_dict = toml.load(toml_file)        
            return toml_dict

        def save_csv(self,filename, df):
            filepath = os.path.join(self.project_path,filename)
            df.to_csv(filepath,index=False)

        def load_csv(self,filename):
            filepath = os.path.join(self.project_path,filename)
            df = pd.read_csv(filepath) 
            return df

        def make_root(self):
            if not os.path.isdir(self.project_path):
                os.mkdir(self.project_path)

        def load_edbo_csv(self):
            self.edbo_df = self.load_csv(self.edbo_filename)
            return

        def save_edbo_csv(self):
            self.save_csv(self.edbo_filename,self.edbo_df)
            return

        def update_edbo_csv(self,update):
            #key = self.update_key
            #dummy0 = self.edbo_df[self.edbo_df[key]=="PENDING"]
            #dummy1 = update[update[key]!="PENDING"]
            key = "priority"
            dummy0 = self.edbo_df[self.edbo_df[key]==0]
            dummy1 = update[update[key]!=0]
            print(dummy0.shape,dummy1.shape)
            self.edbo_df = pd.concat([dummy1,dummy0]).astype({"priority": int})
            return

    def merge_dicts(old,new):
        for key in new:
            old[key] = new[key]
        return old

    class project_information():

        def __init__(self, project_path):
            self.project_path = project_path
            self.project_name = project_path.project_name
            project_info_keys = ["design_variable","design_objective","additional_variable","project_description"]
            self.info = self.load_keys_from_toml("project_info.toml", project_info_keys)
            self.info["name"] = self.project_name

        def update_info(self, update):
            self.info = merge_dicts(self.info, update)


        def load_keys_from_toml(self,toml_file, keys, value=""):
            try:
                data = self.project_path.load_toml(toml_file)
                # check if vars in toml
            except:
                data = {}
                for key in keys:
                    data[key] = value
            return data
    return project_information, project_pathing


@app.cell
def _(mo, new_project_string, project_name_dropdown):
    if project_name_dropdown.value["project_dropdown"] == new_project_string:
        project_name_form = mo.md(
        """

        ## Project Name: {name}

        ---

        ## Project Definition

        ### Describe the Project: {project_description}

        ### Design Components: {design_variable}

        ### Design Objectives: {design_objective}

        ### Additional Variables {additional_variable}

        """   
        ).batch(
            name=mo.ui.text(),
            project_description=mo.ui.text_area(),
            design_variable=mo.ui.text(),
            design_objective=mo.ui.text(),
            additional_variable=mo.ui.text(value="priority"),
           ).form(show_clear_button=True, bordered=True)

    else:
        project_name_form = mo.md("""## {name} """.format(name=project_name_dropdown.value["project_dropdown"]))
    project_name_form
    return (project_name_form,)


@app.cell
def _(
    database_folder_path,
    mo,
    new_project_string,
    project_information,
    project_name_dropdown,
    project_name_form,
    project_pathing,
):
    if project_name_dropdown.value["project_dropdown"] == new_project_string and project_name_form.value: 
        project_path = project_pathing( project_name_form.value["name"], database_folder_path)
        project = project_information(project_path)
        project.update_info(project_name_form.value)
        project_path.make_root()
        project_path.save_toml("project_info.toml",project.info)
    else:
        project_path = project_pathing(project_name_dropdown.value["project_dropdown"], database_folder_path)
        project = project_information(project_path)


    project_overview = mo.md("""

    # Your project: {name}

    ---

    ## Project Definition

    ### Describe the Project: {project_description}

    ### Design Components: {design_variable}

    ### Design Objectives: {design_objective}

    ### Additional Variables {additional_variable}

    """.format(
        name=project.info["name"],
        project_description=project.info["project_description"],
        design_variable=project.info["design_variable"],
        design_objective=project.info["design_objective"],
        additional_variable=project.info["additional_variable"],
    )    )

    project_overview
    return (project_path,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    # Define your Searchspace

    Define you design variables and objective.

    Design variables are ...
    Design objectives are...

    You can also design additional variables you wanna track.
    """
    )
    return


@app.cell
def _(pd, project_path):
    project_form = project_path.load_toml("project_info.toml")

    variable_dict = {}
    variable_table = {}
    variable_list = []
    variables = []
    variable_types = ["design_variable", "design_objective", "additional_variable"]
    save_flag = True
    for x in variable_types:
        ys = project_form[x].split(",")
        variable_dict[x] = ys
        for y in ys:
            if y not in variable_table.keys():
                variable_table[y] = x
                variable_list.append([y,x])
                variables.append(
                    {"name":y,"type":x,"summary":"{y} ({x})".format(x=x,y=y)}
                )
            else:
                print("WARNING: tried to set variable {y} as {x} but it is already initialized as type {xx}".format(y=y,x=x,xx=variable_table[y]))
                save_flag = False
    variable_df = pd.DataFrame(variables)

    #variable_table, variable_list, variable_dict
    return (variable_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Define Your Grid

    you got the following options to define a grid

    ### list:

    A comma separated list of values

    ### linspace:

    typer 3 comma separated values as follows:
    start, stop, number of steps
    Example:
    linspace 1,10,10 leads to: 1,2,3,4,5,6,7,8,9,10

    HINT: the results of your grid definition are dynamically displayed ;)
    """
    )
    return


@app.cell
def _(mo, pd, project_path, variable_df):
    design_variable_df = variable_df[variable_df["type"]=="design_variable"].copy()
    design_variable_no = len(design_variable_df)
    design_variable_options=["list", "linspace"]

    try:
        design_variable_load = pd.DataFrame(project_path.load_toml("design_dict.toml")).T
        design_variable_elements01 = mo.ui.array([ 
            mo.ui.dropdown(options=design_variable_options,value=x["grid_base"]) for i,x in design_variable_load.iterrows()])
        design_variable_elements02 = mo.ui.array([ mo.ui.text(value=x["grid_info"]) for i,x in design_variable_load.iterrows()])    
    except:
        design_variable_elements01 = mo.ui.array([ 
            mo.ui.dropdown(options=design_variable_options,value="linspace") for x in range(design_variable_no)])
        design_variable_elements02 = mo.ui.array([ mo.ui.text() for x in range(design_variable_no)])


    design_variable_form = mo.md(
        f"""
        \n\n
        """
        + "\n\n".join(
            # Iterate over the elements and embed them in markdown
            [
                f"{name} {type0} {type1}"
                for name, (type0,type1) in zip(
                    design_variable_df["summary"], zip(design_variable_elements01,design_variable_elements02)
                )
            ]
        )
    )

    design_variable_form
    return (
        design_variable_df,
        design_variable_elements01,
        design_variable_elements02,
    )


@app.cell
def _(
    design_variable_df,
    design_variable_elements01,
    design_variable_elements02,
    np,
):
    design_variable_df["grid_base"] = design_variable_elements01.value
    design_variable_df["grid_info"] = design_variable_elements02.value

    def get_design_variable_grid(row):
        dummy0 = row["grid_info"].replace("\n","").replace(" ","").split(",")
        try:
            dummy = np.array(dummy0).astype(float)
        except:
            dummy = np.array(dummy0)

        if row["grid_base"] == "linspace":
            try:
                start,stop,num = dummy
                grid = np.linspace(start,stop,num=int(num))
            except:
                grid = "ERROR: linspace not correct"
        else:
            grid = dummy
        return grid

    grid = design_variable_df.apply(get_design_variable_grid, axis=1)
    return (grid,)


@app.cell
def _(mo):
    mo.md(r"""## GRID DISPLAY:""")
    return


@app.cell
def _(design_variable_df, grid, mo):
    grid_show = mo.md(
        f"""
        \n\n
        """
        + "\n\n".join(
            # Iterate over the elements and embed them in markdown
            [
                f"{name} {type}"
                for name, type in zip(
                    design_variable_df["summary"], grid
                )
            ]
        )
    )

    grid_show
    return


@app.cell
def _(design_variable_df, grid):
    design_variable_df["grid"] = grid
    return


@app.cell
def _(mo):
    #project.update_info(project_form.value)

    design_variable_button = mo.ui.run_button(label="Save/Update Design Variables")
    design_variable_button
    return (design_variable_button,)


@app.cell
def _(design_variable_button, design_variable_df, mo, project_path):
    mo.stop( not design_variable_button.value, mo.md("To modify the searchspace update design variables") )
    design_dict = design_variable_df.set_index("name").to_dict(orient='index')
    project_path.save_toml("design_dict.toml",design_dict)
    mo.md("Design variables updated :)")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    # Define Design Objectives

    define the optimization goal of your design objective.
    """
    )
    return


@app.cell
def _(mo, pd, project_path, variable_df):
    design_objective_df = variable_df[variable_df["type"]=="design_objective"].copy()
    design_objective_no = len(design_objective_df)
    design_objective_options=["min", "max"]

    try:
        design_objective_load = pd.DataFrame(project_path.load_toml("objective_dict.toml")).T
        design_objective_elements02 = mo.ui.array([ 
            mo.ui.dropdown(options=design_objective_options,value=x["goal"]) for i,x in design_objective_load.iterrows()])
    except:
        design_objective_elements02 = mo.ui.array([ 
            mo.ui.dropdown(options=design_objective_options,value="max") for x in range(design_objective_no)])

    design_objective_form = mo.md(
        f"""
        \n\n
        """
        + "\n\n".join(
            # Iterate over the elements and embed them in markdown
            [
                f"{name} {type0}"
                for name, type0 in zip(
                    design_objective_df["summary"], design_objective_elements02
                )
            ]
        )
    )

    design_objective_form
    return design_objective_df, design_objective_elements02


@app.cell
def _(mo):
    design_objective_button = mo.ui.run_button(label="Save/Update Design Objectives")
    design_objective_button
    return (design_objective_button,)


@app.cell
def _(
    design_objective_button,
    design_objective_df,
    design_objective_elements02,
    mo,
    project_path,
):
    objective_dict = design_objective_df.set_index("name").to_dict(orient='index')
    mo.stop( not design_objective_button.value, mo.md("To modify your search update design objectives") )
    design_objective_df["goal"] = design_objective_elements02.value
    objective_dict = design_objective_df.set_index("name").to_dict(orient='index')
    project_path.save_toml("objective_dict.toml",objective_dict)
    mo.md("Design objectives updated :)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    # RUN EDBO+

    Finally, we can run EDBO+ based on our definitions but first we may build our reaction scope:
    """
    )
    return


@app.cell
def _(mo, os, project_path):
    if os.path.isfile(project_path.edbo_filepath):
        button_form = (
            mo.md(
                '''
            ## Reaction Scope Already Exists

            {btn}

            '''
            )
            .batch(
                btn=mo.ui.run_button(label="Rebuild Scope"),
            )
        )
    else:
        button_form = (
            mo.md(
                '''
            ## Build Reaction Scope

            First, you need to build the reaction scope:

            {btn}

            '''
            )
            .batch(
                btn=mo.ui.run_button(label="Build Scope"),
            )
        )

    button_form
    return (button_form,)


@app.cell
def _(
    EDBOplus,
    button_form,
    design_objective_df,
    mo,
    np,
    os,
    pd,
    project_path,
    reaction_components,
    shutil,
):
    mo.stop( not button_form.value["btn"], mo.md("DIDN'T (re)build scope") )

    if os.path.exists(project_path.edbo_filepath):
        message = "found and used old optimization/experiments/scope"
        # copy current scope to save results
        dest = os.path.join(project_path.project_path,"my_optimization_backup.csv")
        shutil.move(project_path.edbo_filepath,dest)

        EDBOplus().generate_reaction_scope(
            components=reaction_components,
            directory=project_path.project_path,
            filename=project_path.edbo_filename,
            check_overwrite=False
        )    

        # load backup scope
        backup_scope = pd.read_csv(dest)
        if "priority" in backup_scope.keys():
            backup_scope = backup_scope[backup_scope["priority"]==-1]

            # load new scope
            new_scope = pd.read_csv(project_path.edbo_filepath)
            new_scope["priority"] = 0
            new_scope[list(np.atleast_1d(design_objective_df["name"]))] = "PENDING"

            # Concatenate scopes, ignore original indexes
            combined = pd.concat([backup_scope,new_scope], ignore_index=True)

            # Drop duplicates based on reaction_components, keeping the row from backup_scope if present
            result = combined.drop_duplicates(subset=reaction_components, keep='first').reset_index(drop=True)
            result["priority"][result["priority"] != -1] = 0
            result.to_csv(project_path.edbo_filepath, index=False)        

    else:
        message = "NEW NEW NEW"
        EDBOplus().generate_reaction_scope(
            components=reaction_components,
            directory=project_path.project_path,
            filename=project_path.edbo_filename,
            check_overwrite=False
        )          
        scope = pd.read_csv(project_path.edbo_filepath)
        scope["priority"] = 0
        scope[list(np.atleast_1d(design_objective_df["name"]))] = "PENDING"
        result.to_csv(project_path.edbo_filepath, index=False)    
    

    #    EDBOplus().run(
    #        filename=project_path.edbo_filename,  # Previously generated scope.
    #        directory=project_path.project_path,
    #        objectives=list(np.atleast_1d(design_objective_df["name"])),  # Objectives to be optimized.
    #        objective_mode=list(np.atleast_1d(design_objective_df["goal"])),  # Maximize yield and ee but minimize side_product.
    #        batch=3,  # Number of experiments in parallel that we want to perform in this round.
    #        columns_features='all', # features to be included in the model.
    #        init_sampling_method='cvt'  # initialization method.
    #    )

    mo.md("""
        DID (re)build scope :)
        ---
        {message}
        ---
        """.format(message=message)) 
    return


@app.cell
def _(mo):
    bayes_button = mo.ui.run_button(label="Run EDBO+")
    bayes_button
    return (bayes_button,)


@app.cell
def _(
    EDBOplus,
    bayes_button,
    design_objective_df,
    design_variable_df,
    mo,
    np,
    project_path,
):
    reaction_components = design_variable_df[["name","grid"]].set_index("name").to_dict()["grid"]

    mo.stop( not bayes_button.value, mo.md("press Run EDBO+ button to run EDBO+") )

    EDBOplus().run(
            filename=project_path.edbo_filename,  # Previously generated scope.
            directory=project_path.project_path,
            objectives=list(np.atleast_1d(design_objective_df["name"])),  # Objectives to be optimized.
            objective_mode=list(np.atleast_1d(design_objective_df["goal"])),  # Maximize yield and ee but minimize side_product.
            batch=3,  # Number of experiments in parallel that we want to perform in this round.
            columns_features='all', # features to be included in the model.
            init_sampling_method='cvt'  # initialization method.
    )

    project_path.load_edbo_csv() # self.edbo_df
    df_predictions = project_path.edbo_df[project_path.edbo_df["priority"]!=0]
    df_pred_show = df_predictions.style.background_gradient(subset=['priority'], cmap='plasma')

    df_pred_show

    return df_predictions, reaction_components


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## EDBO+ Results
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    # Now you can do your Experiments :)


    ---


    ## ...and add the results (double click to change, dont forget Priority):
    """
    )
    return


@app.cell
def _(bayes_button, df_predictions, mo):
    mo.stop( not bayes_button.value, mo.md("After running EDBO+, new experiments can be added here :)") )
    df_predictions_editor = mo.ui.data_editor(data=df_predictions, label="Edit Data")
    df_predictions_editor
    return (df_predictions_editor,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## Before saving you can check again:
    """
    )
    return


@app.cell
def _(bayes_button, df_predictions_editor, mo, project_path):
    disable_save = True
    mo.stop( not bayes_button.value, mo.md("After running EDBO+, added experiments are shown here :)") )
    disable_save = disable_save
    project_path.update_edbo_csv(df_predictions_editor.value)
    df_edbo_show = project_path.edbo_df.reset_index()

    df_edbo_show
    return (disable_save,)


@app.cell
def _(mo):
    mo.md(r"""## save, save, save....""")
    return


@app.cell
def _(disable_save, mo):
    save_experiment_button = mo.ui.run_button(label="Save/Update Experiments",disabled=disable_save)
    save_experiment_button
    return


@app.cell
def _(bayes_button, df_predictions_editor, mo, project_path):
    mo.stop( not bayes_button.value, mo.md("Run EDBO+ to be ale to save ur new experiments :)") )
    project_path.update_edbo_csv(df_predictions_editor.value)
    project_path.save_edbo_csv()
    mo.md("Results saved :)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
