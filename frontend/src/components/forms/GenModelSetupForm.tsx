import { ChangeEvent, FC, useEffect, useState } from 'react';
import { useGetGenModels } from '../../core/api';
import { GenModel, GenModelAPI } from '../../types';

export const GenModelSetupForm: FC = () => {
  const [availableAPIs, setAvailableAPIs] = useState<GenModelAPI[]>([]);
  const [apiModels, setApiModels] = useState<GenModel[]>([]);
  const [selectedAPI, setSelectedAPI] = useState<GenModelAPI>(availableAPIs[0]);
  const { models } = useGetGenModels();
  useEffect(() => {
    const fetchModels = async () => {
      setAvailableAPIs(await models());
    };
    fetchModels();
  }, [models]);
  useEffect(() => {
    console.log('set to ', availableAPIs[0]);
    setSelectedAPI(availableAPIs[0]);
  }, [availableAPIs]);

  const onAPIChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(e.target.value);
    if (index >= availableAPIs.length) {
      throw new Error(`Invalid index choice: ${index}`);
    }
    setSelectedAPI(availableAPIs[index]);
    setApiModels(availableAPIs[index].models);
  };

  return (
    <form className="d-flex flex-wrap gap-3">
      <div className="form-floating col-5">
        <select className="form-select" id="api" defaultValue={0} onChange={onAPIChange}>
          {availableAPIs.map((api, index) => (
            <option key={index} value={index}>
              {api.name}
            </option>
          ))}
        </select>
        <label htmlFor="api">API </label>
      </div>
      {(() => {
        const inputs = [];
        if (selectedAPI !== undefined) {
          if (selectedAPI.name === 'OpenAI')
            inputs.push(
              <div className="form-floating col-5">
                <select className="form-control" id="model">
                  {selectedAPI.models.map((model) => (
                    <option key={model.id}>{model.name}</option>
                  ))}
                </select>
                {/* mt-0 is needed because a rule in the CSS file add a margin to .form-label. Maybe it should be removed*/}
                <label htmlFor="model" className="mt-0">
                  Model
                </label>
              </div>,
            );
          else
            inputs.push(
              <div className="form-floating col-5">
                <input
                  type="text"
                  id="model"
                  className="form-control"
                  placeholder="ID of the model"
                />
                <label htmlFor="model" className="mt-0">
                  Model
                </label>
              </div>,
            );
          inputs.push(
            <div className="form-floating col-5">
              <input
                type="text"
                id="endpoint"
                className="form-control"
                placeholder="enter the url of the endpoint"
              />
              <label htmlFor="endpoint" className="mt-0">
                Endpoint
              </label>
            </div>,
          );

          if (selectedAPI.name !== 'Ollama')
            inputs.push(
              <div className="form-floating col-5">
                <input
                  type="text"
                  id="credentials"
                  className="form-control"
                  placeholder="API key"
                />
                <label htmlFor="credentials" className="mt-0">
                  API Credentials
                </label>
              </div>,
            );
        }
        return inputs;
      })()}
    </form>
  );
};
