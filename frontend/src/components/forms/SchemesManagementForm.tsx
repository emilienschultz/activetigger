import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useAddScheme, useDeleteScheme } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { SchemeModel } from '../../types';

interface SchemesManagementProps {
  available_schemes: string[];
  projectSlug: string;
}

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<SchemesManagementProps> = ({
  available_schemes,
  projectSlug,
}) => {
  // get element from the context
  const {
    appContext: { currentScheme, reFetchCurrentProject },
    setAppContext,
  } = useAppContext();

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, currentScheme || null);

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const handleIconClick = () => {
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // put the current scheme in the context on change
  const handleSelectScheme = (event: any) => {
    setAppContext((state) => ({
      ...state,
      currentScheme: event.target.value,
    }));
    notify({ type: 'success', message: 'Scheme selected' });
  };

  // action to create the new scheme
  const createNewScheme: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      await addScheme(formData.name);
      if (reFetchCurrentProject) reFetchCurrentProject();
      notify({ type: 'success', message: `Scheme ${formData.name} created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // action to delete scheme
  const deleteSelectedScheme = async () => {
    //TODO: try catch and throw
    await deleteScheme();
    if (reFetchCurrentProject) reFetchCurrentProject();
  };
  return (
    <div className="container-fluid">
      <div className="row">
        <label htmlFor="scheme-selected">Available schemes</label>
        <div>
          <select
            id="scheme-selected"
            className="form-select-lg col-3 mb-3"
            onChange={handleSelectScheme}
            value={currentScheme ? currentScheme : ''}
          >
            <option></option> {/*empty possibility*/}
            {available_schemes.map((element) => (
              <option key={element} value={element} selected={element === currentScheme}>
                {element}
              </option>
            ))}{' '}
          </select>
          <button onClick={deleteSelectedScheme} className="btn btn p-0">
            <MdOutlineDeleteOutline size={30} />
          </button>
          <button onClick={handleIconClick} className="btn p-0">
            <IoIosAddCircle size={30} />
          </button>
        </div>
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewScheme && (
            <div className="row">
              <form onSubmit={handleSubmit(createNewScheme)}>
                <div className="secondary-panel col-4">
                  <label className="form-label" htmlFor="scheme_name">
                    New scheme
                  </label>
                  <input
                    className="form-control"
                    id="scheme_name"
                    type="text"
                    {...register('name')}
                  />
                  <button className="btn btn-primary btn-validation">Create</button>
                </div>
              </form>
            </div>
          )
        }
      </div>
    </div>
  );
};
