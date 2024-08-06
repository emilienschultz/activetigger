import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useAddFeature, useDeleteFeature } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { FeatureModelExtended } from '../../types';

interface FeaturesManagementProps {
  availableFeatures: string[];
  possibleFeatures: { [key: string]: { [key: string]: any } };
  projectSlug: string;
  reFetchProject: () => void;
}

/* Manage schemes
 * Select ; Delete ; Add
 */

export const FeaturesManagement: FC<FeaturesManagementProps> = ({
  availableFeatures,
  possibleFeatures,
  projectSlug,
  reFetchProject,
}) => {
  // hooks to use the objets
  const { register, handleSubmit } = useForm<FeatureModelExtended>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addFeature = useAddFeature(projectSlug);
  const deleteFeature = useDeleteFeature(projectSlug);

  // state for displaying the new scheme menu
  const [showCreateNewFeature, setShowCreateNewFeature] = useState(false);
  const handleAddIconClick = () => {
    setShowCreateNewFeature(!showCreateNewFeature);
  };

  // state for the current selected feature
  const [getSelectedScheme, setSelectedScheme] = useState(null);

  // manage selected feature
  const handleSelectFeature = (event: any) => {
    setSelectedScheme(event.target.value);
  };

  // state for the type of feature to create
  const [selectedFeatureToCreate, setFeatureToCreate] = useState('');

  // action to create the new scheme
  const createNewFeature: SubmitHandler<FeatureModelExtended> = async (formData) => {
    try {
      addFeature(formData.type, formData.name, formData.parameters);
      notify({ type: 'success', message: `Feature created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    reFetchProject();
    setShowCreateNewFeature(!showCreateNewFeature);
  };

  // action to delete feature
  const deleteSelectedFeature = async () => {
    //TODO: try catch and throw
    await deleteFeature(getSelectedScheme);
    reFetchProject();
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <label className="form-label" htmlFor="existing-feature-selected">
          Existing features
        </label>
        <div>
          <select
            id="existing-feature-selected"
            className="form-select-lg col-3 mb-3"
            onChange={handleSelectFeature}
          >
            <option></option> {/*empty possibility*/}
            {availableFeatures.map((element) => (
              <option key={element} value={element}>
                {element}
              </option>
            ))}{' '}
          </select>
          {
            <button className="btn btn p-0" onClick={deleteSelectedFeature}>
              <MdOutlineDeleteOutline size={30} />
            </button>
          }
          <button onClick={handleAddIconClick} className="btn p-0">
            <IoIosAddCircle size={30} />
          </button>
        </div>
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewFeature && (
            <div className="row">
              <form onSubmit={handleSubmit(createNewFeature)}>
                <div className="secondary-panel col-4">
                  <label className="form-label" htmlFor="newFeature">
                    Add new feature
                  </label>
                  <select
                    className="form-control"
                    id="newFeature"
                    {...register('type')}
                    onChange={(event) => {
                      setFeatureToCreate(event.target.value);
                    }}
                  >
                    <option></option>
                    {Object.keys(possibleFeatures).map((element) => (
                      <option key={element} value={element}>
                        {element}
                      </option>
                    ))}{' '}
                  </select>
                  <button className="btn btn-primary btn-validation">Create</button>

                  {selectedFeatureToCreate === 'dfm' && (
                    <div>
                      <div>
                        <label htmlFor="dfm_tfidf">TF-IDF</label>
                        <select id="dfm_tfidf" {...register('parameters.dfm_tfidf')}>
                          <option>True</option>
                          <option>False</option>
                        </select>
                      </div>
                      <div>
                        <label htmlFor="dfm_ngrams">Ngrams</label>
                        <input
                          type="number"
                          id="dfm_ngrams"
                          value={1}
                          {...register('parameters.dfm_ngrams')}
                        />
                      </div>
                      <div>
                        <label htmlFor="dfm_min_term_freq">Min term freq</label>
                        <input
                          type="number"
                          id="dfm_min_term_freq"
                          value={5}
                          {...register('parameters.dfm_min_term_freq')}
                        />
                      </div>
                      <div>
                        <label htmlFor="dfm_max_term_freq">Max term freq</label>
                        <input
                          type="number"
                          id="dfm_max_term_freq"
                          value={100}
                          {...register('parameters.dfm_max_term_freq')}
                        />
                      </div>
                      <div>
                        <label htmlFor="dfm_norm">Norm</label>
                        <select id="dfm_norm" {...register('parameters.dfm_norm')}>
                          <option>False</option>
                          <option>True</option>
                        </select>
                      </div>
                      <div>
                        <label htmlFor="dfm_log">Log</label>
                        <select id="dfm_log" {...register('parameters.dfm_log')}>
                          <option>False</option>
                          <option>True</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              </form>
            </div>
          )
        }
      </div>
    </div>
  );
};
